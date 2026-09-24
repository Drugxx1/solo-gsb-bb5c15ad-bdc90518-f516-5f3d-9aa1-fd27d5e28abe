"""跨事件概率校准报告：把揭示原文解析为 [0,1] 概率并做十进制精确校准统计。

纯函数模块：不依赖数据库与 HTTP，可独立单测。

概率文本语法（严格匹配，不允许空白、符号、指数、下划线、Infinity/NaN 等
宽松形式）：

    probability := ( 数字+ ["." 数字*] | "." 数字+ )

与数值区间报告的有限十进制数不同：概率**不接受**正负号（``+`` / ``-``）
与指数记法（``e`` / ``E``），也不接受任何空白；解析得到的十进制数还必须
落在闭区间 [0, 1]。``"0"``、``"1"``、``"0.5"``、``".25"``、``"1.00"``
合法；``"+0.5"``、``"1e-1"``、``" 0.5"``、``"1.00001"``、``"-0"`` 非法。

校准指标（Brier 均分与分箱）在量化到 6 位小数之前全程保持精确：平方
误差用整数系数运算得到精确 ``decimal.Decimal``（``** 2`` 会按默认上下文
28 位精度提前舍入，例如 50 位概率平方后丢末位会使量化结果出错）；跨事件
均分与分箱均值 / 正例频率以精确 ``fractions.Fraction`` 保存（除法一般
不终止，不能物化成 Decimal）。对外数值统一四舍六入五成双
（ROUND_HALF_EVEN）量化到 6 位小数，量化同样用整数算术完成，不经过任何
浮点或上下文精度舍入；因此无论输入概率有多少位小数、输入顺序如何，结果
都与精确十进制计算一致。
"""

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from fractions import Fraction
from uuid import UUID

# [0,1] 概率的严格语法：无空白、无符号、无指数；显式排除 Decimal 构造器
# 宽容接受的下划线、Infinity 与 NaN 等形式（仅 ASCII 数字；正则的 \d 会
# 匹配全角数字等其他 Unicode 数字字符，这里显式用 [0-9]）
PROBABILITY_PATTERN = re.compile(r"^([0-9]+(\.[0-9]*)?|\.[0-9]+)$")

# 无效原因
INVALID_FORMAT = "INVALID_FORMAT"  # 不是无空白 / 无符号 / 无指数的十进制数
OUT_OF_RANGE = "OUT_OF_RANGE"  # 数值不在闭区间 [0, 1]

# 对外数值的统一精度：四舍六入五成双，6 位小数
SIX_PLACES = Decimal("0.000001")

ZERO = Decimal(0)
ONE = Decimal(1)

# 十个分箱的左边界（最后一个为 [0.9, 1]，右端点 1 归入第 10 箱）
BIN_COUNT = 10
BIN_LOWER_BOUNDS = [Decimal(i) / Decimal(10) for i in range(BIN_COUNT)]


# 量化到 6 位小数时的放大倍数
_QUANT_SCALE = 1_000_000


def quantize6(value: Decimal | Fraction) -> Decimal:
    """四舍六入五成双量化到 6 位小数（数值对外输出前的统一处理）。

    - ``Decimal`` 输入为**精确**值（如整数系数算得的平方误差），直接按
      ROUND_HALF_EVEN 量化（``quantize`` 不按上下文精度舍入结果）；
    - ``Fraction`` 输入为精确有理数（如 Brier 均分、分箱均值），转用整数
      算术完成五成双判定（见 ``_quantize6_ratio``），不经过有限精度除法。
    """
    if isinstance(value, Fraction):
        return _quantize6_ratio(value.numerator, value.denominator)
    return value.quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)


def _quantize6_ratio(numerator: int, denominator: int) -> Decimal:
    """把非负有理数 ``numerator / denominator`` 精确量化到 6 位小数。

    全程整数算术：``q = floor(10^6 · n / d)`` 为向下取整结果，余数 ``r``
    满足真实值 = (q + r/d) · 10^-6；``2r > d`` 向上取整，``2r < d``
    向下取整，恰为一半（``2r == d``）时五成双（``q`` 奇则进一、偶则
    不进）。结果用 (符号, 数字元组, 指数) 三元组直接构造，绕过 Decimal
    上下文精度。调用方保证数值非负（概率、平方误差与频率均在 [0, 1]）。
    """
    quotient, remainder = divmod(numerator * _QUANT_SCALE, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    digits = tuple(int(d) for d in str(quotient))
    return Decimal((0, digits, -6))


def _signed_coefficient_exponent(value: Decimal) -> tuple[int, int]:
    """取有限 Decimal 的 (带符号整数系数, 指数)（负零归一为 0）。

    与 intervals.py 中同名助手一致：整数系数运算的入口。
    """
    sign, digits, exponent = value.as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return (-coefficient if sign else coefficient), exponent


def _decimal_from_coefficient(coefficient: int, exponent: int) -> Decimal:
    """用 (符号, 数字元组, 指数) 直接构造 Decimal，绕过上下文精度舍入。

    ``Decimal(coefficient).scaleb(exponent)`` 与普通算术运算都会按当前
    上下文精度（默认 28 位有效数字）舍入系数；元组构造则对任意长度
    系数都保持精确。
    """
    if coefficient == 0:
        digits = (0,)
    else:
        digits = tuple(int(d) for d in str(abs(coefficient)))
    return Decimal((1 if coefficient < 0 else 0, digits, exponent))


def exact_squared_error(probability: Decimal, outcome: int) -> Decimal:
    """精确计算平方误差 ``(probability - outcome)^2``（outcome ∈ {0, 1}）。

    Decimal 算术按当前上下文精度（默认 28 位有效数字）舍入；高精度概率
    直接 ``** 2`` 会提前丢末位（如
    ``0.5000004999…833`` 的精确平方是 ``0.2500005000…5778…``，量化到
    6 位应为 ``0.250001``，而 28 位舍入后的平方恰为 ``0.2500005``，
    五成双量化误得 ``0.250000``）。

    这里改用整数系数：严格语法下概率不含指数记法，必可写成
    ``c × 10^e`` 且 ``e <= 0``，于是

        outcome = 0: (p-0)^2 = c^2 · 10^(2e)
        outcome = 1: (p-1)^2 = (10^(-e) - c)^2 · 10^(2e)   （0 <= p <= 1）

    整数平方精确无舍入，再按 2 倍指数直接构造 Decimal。
    """
    coefficient, exponent = _signed_coefficient_exponent(probability)
    if outcome == 1:
        coefficient = 10 ** (-exponent) - coefficient
    return _decimal_from_coefficient(coefficient * coefficient, 2 * exponent)


def parse_probability(text: str) -> tuple[Decimal | None, str | None]:
    """把揭示原文解析为 [0, 1] 内的精确十进制概率。

    返回 (概率, None) 或 (None, 原因)；原因取 INVALID_FORMAT（不是无空白 /
    无符号 / 无指数的十进制数）或 OUT_OF_RANGE（数值超出 [0, 1]）。
    """
    if PROBABILITY_PATTERN.fullmatch(text) is None:
        return None, INVALID_FORMAT
    try:
        value = Decimal(text)
    except InvalidOperation:
        # 语法合法但位数超出 Decimal 当前上下文可表示范围
        return None, INVALID_FORMAT
    # is_finite 为防御性检查：上述语法已排除 Infinity/NaN；范围检查给出
    # 明确的 OUT_OF_RANGE 原因
    if not value.is_finite():
        return None, INVALID_FORMAT
    if value < ZERO or value > ONE:
        return None, OUT_OF_RANGE
    return value, None


@dataclass(frozen=True)
class CalibrationPrediction:
    """一条有效概率预测（原文已通过严格解析并落在 [0, 1]）。"""

    event_id: UUID
    commitment_id: UUID
    committed_at: datetime
    original_text: str  # 揭示原文逐字保留
    probability: Decimal  # 精确十进制概率（[0, 1]）
    outcome: int  # 结算标准结果：0 | 1
    squared_error: Decimal  # (probability - outcome)^2，精确十进制


@dataclass(frozen=True)
class CalibrationInvalid:
    """一条无法作为概率使用的已揭示原文。"""

    event_id: UUID
    commitment_id: UUID
    committed_at: datetime
    original_text: str
    reason: str  # INVALID_FORMAT | OUT_OF_RANGE


@dataclass(frozen=True)
class CalibrationBin:
    """一个概率分箱的汇总（空样本时 mean / positive_rate 为 None）。"""

    index: int  # 0～9
    lower: Decimal  # 左边界（含）
    upper: Decimal  # 右边界（不含；最后一箱右端点 1 含）
    count: int
    mean: Fraction | None  # 箱内概率均值（精确有理数；空样本为 None）
    positive_rate: Fraction | None  # 箱内实际为 1 的频率（精确有理数；空样本为 None）


@dataclass(frozen=True)
class CalibrationReport:
    """校准报告的确定性结果（排序与分箱均已完成，数值仍为精确值）。"""

    predictions: list[CalibrationPrediction]  # 按 (事件序列, 提交时间, 承诺 ID) 升序
    invalid: list[CalibrationInvalid]  # 同上
    bins: list[CalibrationBin]  # 固定 10 箱，按下标 0～9 升序
    brier_mean: Fraction | None  # 全部有效预测平方误差均值（精确有理数）；无有效预测为 None
    unrevealed_count: int  # 这些事件中从未揭示的承诺总数


@dataclass(frozen=True)
class CalibrationSource:
    """一条已揭示记录的原始输入（由路由从数据库组装）。"""

    event_id: UUID
    commitment_id: UUID
    committed_at: datetime
    original_text: str
    outcome: int  # 所属事件的结算标准结果：0 | 1


def bin_index_for(probability: Decimal) -> int:
    """按确定的边界归属返回分箱下标：[0,0.1), …, [0.8,0.9), [0.9,1]。

    边界 k/10（k=1..9）归入第 k 箱（左闭右开，边界属于右侧箱）；端点 1
    归入最后一箱。全程精确十进制比较。
    """
    index = 0
    for i, lower in enumerate(BIN_LOWER_BOUNDS):
        if probability >= lower:
            index = i
    return index


def build_calibration_report(
    sources: list[CalibrationSource],
    unrevealed_count: int,
) -> CalibrationReport:
    """对跨事件的已揭示原文做概率解析、Brier 计分与分箱汇总。

    - sources 输入顺序任意；输出中有效 / 无效条目均按
      (event_id, committed_at, commitment_id) 稳定排序（UUID 取 128 位
      整数比较）；
    - 每条有效预测的平方误差 = (probability - outcome)^2，用整数系数
      精确计算，不受 Decimal 默认上下文 28 位精度影响；
    - Brier 均分 = 有效预测平方误差之和 / 有效预测数，以精确有理数保存
      （无有效预测为 None）；
    - 十个分箱固定返回（空样本均值与正例频率均为 None，count 为 0），
      箱内均值与频率同样为精确有理数。

    有理数累加与输入顺序无关（整数分子分母的运算结果唯一），最终输出
    前才由 quantize6 五成双量化到 6 位小数。
    """
    predictions: list[CalibrationPrediction] = []
    invalid: list[CalibrationInvalid] = []
    for source in sources:
        probability, reason = parse_probability(source.original_text)
        if probability is None:
            invalid.append(
                CalibrationInvalid(
                    event_id=source.event_id,
                    commitment_id=source.commitment_id,
                    committed_at=source.committed_at,
                    original_text=source.original_text,
                    reason=reason,
                )
            )
            continue
        squared_error = exact_squared_error(probability, source.outcome)
        predictions.append(
            CalibrationPrediction(
                event_id=source.event_id,
                commitment_id=source.commitment_id,
                committed_at=source.committed_at,
                original_text=source.original_text,
                probability=probability,
                outcome=source.outcome,
                squared_error=squared_error,
            )
        )

    # 与其他报告一致：时间相同时按承诺 ID（UUID 128 位整数）决胜；
    # 跨事件再先按事件 ID 分组排序
    predictions.sort(
        key=lambda p: (p.event_id.int, p.committed_at, p.commitment_id.int)
    )
    invalid.sort(key=lambda p: (p.event_id.int, p.committed_at, p.commitment_id.int))

    brier_mean: Fraction | None = None
    if predictions:
        # Fraction(Decimal) 为精确转换；有理数求和 / 除法永不终止也不丢位
        total = sum((Fraction(p.squared_error) for p in predictions), Fraction(0))
        brier_mean = total / len(predictions)

    # 固定 10 个分箱：第 0～8 箱为 [k/10, (k+1)/10)，第 9 箱为 [0.9, 1]
    bin_sums: list[Fraction] = [Fraction(0)] * BIN_COUNT
    bin_positive: list[int] = [0] * BIN_COUNT
    bin_counts: list[int] = [0] * BIN_COUNT
    for p in predictions:
        i = bin_index_for(p.probability)
        bin_counts[i] += 1
        bin_sums[i] += Fraction(p.probability)
        if p.outcome == 1:
            bin_positive[i] += 1

    bins: list[CalibrationBin] = []
    for i in range(BIN_COUNT):
        lower = BIN_LOWER_BOUNDS[i]
        upper = ONE if i == BIN_COUNT - 1 else BIN_LOWER_BOUNDS[i + 1]
        count = bin_counts[i]
        if count == 0:
            mean: Fraction | None = None
            positive_rate: Fraction | None = None
        else:
            mean = bin_sums[i] / count
            positive_rate = Fraction(bin_positive[i], count)
        bins.append(
            CalibrationBin(
                index=i,
                lower=lower,
                upper=upper,
                count=count,
                mean=mean,
                positive_rate=positive_rate,
            )
        )

    return CalibrationReport(
        predictions=predictions,
        invalid=invalid,
        bins=bins,
        brier_mean=brier_mean,
        unrevealed_count=unrevealed_count,
    )
