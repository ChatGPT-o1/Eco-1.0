"""Hand-written annotations over the base model's own MBPP drafts.

REPAIRS: for a failed draft -> one-line root-cause FAULT, a PIVOT, and either
         `edits` (exact old->new replacements on the draft: the surgical fix) or
         `code` (full rewrite when the approach itself is wrong).
         build_data.py executes every fix against the tests and drops any that fail.
FRAMING: one grounded sentence said before first-pass code.
SKIP:    failed drafts deliberately left out (spec only knowable from hidden tests, etc).
"""

REPAIRS = [
    dict(id="mbpp/960",
         fault="The base cases put f(2) at 2, so the sequence runs a step ahead and f(4) comes out 5 instead of 3.",
         pivot="This count is Fibonacci seeded with f(1) = f(2) = 1; fix the seeds and keep the loop.",
         edits=[["    if n == 2:\n        return 2", "    if n == 2:\n        return 1"],
                ["a, b = 1, 2", "a, b = 1, 1"]]),
    dict(id="mbpp/765",
         fault="`is_polite` tests divisibility by 2 and 3, but the test wants the nth polite number "
               "(`is_polite(7) == 11`), and polite numbers are exactly the integers that aren't powers of two.",
         pivot="Make `is_polite(n)` count upward from 3, skipping powers of two (`k & (k - 1) == 0`), and return the nth hit.",
         code="def is_polite(n):\n    count, k = 0, 2\n    while True:\n        k += 1\n        if k & (k - 1):  # not a power of two\n"
              "            count += 1\n            if count == n:\n                return k\n"),
    dict(id="mbpp/608",
         fault="The test calls `bell_Number` but the function is `bell_number`, and the recurrence is off too: "
               "`B[k] * (i - 1) // (i - k)` is not C(i-1, k).",
         pivot="Rename it and build the Bell triangle, where each row starts with the previous row's last value.",
         code="def bell_Number(n):\n    row = [1]\n    for _ in range(n):\n        nxt = [row[-1]]\n        for x in row:\n"
              "            nxt.append(nxt[-1] + x)\n        row = nxt\n    return row[0]\n"),
    dict(id="mbpp/918",
         fault="The arguments are read as (coins, target, max_coins), but `coin_change([1, 2, 3], 3, 4) == 4` is "
               "(coins, number of coins, amount), and repeating the DP `max_coins` times overcounts to 40.",
         pivot="Treat the third argument as the amount and run the standard single-pass coin DP over the first m coins.",
         code="def coin_change(coins, m, n):\n    dp = [1] + [0] * n\n    for c in coins[:m]:\n        for i in range(c, n + 1):\n"
              "            dp[i] += dp[i - c]\n    return dp[n]\n"),
    dict(id="mbpp/912",
         fault="No code came out: the reply spent its whole budget guessing formulas. `lobb_num(n, m)` is the Lobb number "
               "L(m, n) = (2m + 1) / (m + n + 1) · C(2n, m + n), which gives 35 for (5, 3).",
         pivot="Implement that closed form directly with `math.comb`.",
         code="from math import comb\n\n\ndef lobb_num(n, m):\n    return (2 * m + 1) * comb(2 * n, m + n) / (m + n + 1)\n"),
    dict(id="mbpp/638",
         fault="The coefficients and the w² / w³ terms aren't the wind chill formula, and the arguments are swapped: "
               "the call is `wind_chill(velocity_kmh, temp_c)`.",
         pivot="Use the standard index 13.12 + 0.6215·T − 11.37·V^0.16 + 0.3965·T·V^0.16, rounded to an int.",
         code="def wind_chill(v, t):\n    w = v ** 0.16\n    return round(13.12 + 0.6215 * t - 11.37 * w + 0.3965 * t * w)\n"),
    dict(id="mbpp/814",
         fault="The two arguments are the diagonals, not equal sides plus an angle, so the equal-sides check rejects "
               "valid input; the area is p·q / 2 (10 × 20 / 2 = 100).",
         pivot="Drop the side/angle logic and return half the product of the diagonals.",
         code="def rombus_area(p, q):\n    return p * q / 2\n"),
    dict(id="mbpp/641",
         fault="`9 * n * (n - 1) // 2` isn't the nonagonal formula; the nth nonagonal number is n(7n − 5) / 2, "
               "which gives 325 for n = 10.",
         pivot="Swap in the correct closed form.",
         edits=[["return 9 * n * (n - 1) // 2", "return n * (7 * n - 5) // 2"]]),
    dict(id="mbpp/752",
         fault="The closed form has the sign flipped: J(n) = (2ⁿ − (−1)ⁿ) / 3, so adding `(-1) ** n` gives 10 instead of 11 for n = 5.",
         pivot="Subtract the alternating term.",
         edits=[["return (2 ** n + ((-1) ** n)) // 3", "return (2 ** n - (-1) ** n) // 3"]]),
    dict(id="mbpp/910",
         fault="The parameters are read as (year, month, day) but the calls pass (month, day, year), possibly as strings, "
               "and the month table is padded and missing December, so valid dates are rejected.",
         pivot="Take (month, day, year), coerce to int, and let `datetime.date` do the validation.",
         code="import datetime\n\n\ndef check_date(m, d, y):\n    try:\n        datetime.date(int(y), int(m), int(d))\n"
              "        return True\n    except ValueError:\n        return False\n"),
    dict(id="mbpp/934",
         fault="The double loop multiplies unrelated flattened binomials and powers; the Delannoy number is one sum, "
               "D(m, n) = Σₖ C(m, k)·C(n, k)·2ᵏ.",
         pivot="Replace the loops with that sum over k up to min(m, n).",
         code="from math import comb\n\n\ndef dealnnoy_num(m, n):\n"
              "    return sum(comb(m, k) * comb(n, k) * 2 ** k for k in range(min(m, n) + 1))\n"),
    dict(id="mbpp/952",
         fault="The Lucas-theorem term `(ni - ri)! * (p - diff)! * (p - 1)^ri` isn't a valid identity, and p isn't even "
               "prime in the tests (`p = 14`), so the modular trick is unsound.",
         pivot="Compute the exact binomial with `math.comb` and reduce mod p once; Python ints keep it exact.",
         code="from math import comb\n\n\ndef nCr_mod_p(n, r, p):\n    return comb(n, r) % p\n"),
    dict(id="mbpp/926",
         fault="The reply never produced code: it treated rencontres numbers as plain binomials and stalled when 21 ≠ 924. "
               "D(n, k) counts permutations with exactly k fixed points, C(n, k) · !(n − k), and C(7, 2) · !5 = 21 · 44 = 924.",
         pivot="Compute the derangement count iteratively and multiply by `comb(n, k)`.",
         code="from math import comb\n\n\ndef rencontres_number(n, m):\n    def derangements(k):\n        if k == 0:\n"
              "            return 1\n        a, b = 1, 0  # !0, !1\n        for i in range(2, k + 1):\n"
              "            a, b = b, (i - 1) * (a + b)\n        return b\n\n    return comb(n, m) * derangements(n - m)\n"),
    dict(id="mbpp/582",
         fault="The test calls `my_dict`, but the function is defined as `is_empty`, so the name doesn't exist.",
         pivot="Name it `my_dict`; `not d` already covers the empty check.",
         edits=[["def is_empty(d):", "def my_dict(d):"], ["return bool(d) == False", "return not d"]]),
]

FRAMING = {
    'mbpp/583': 'Closed form through factorials: C(2n, n) / (n + 1), kept in exact integer math.',
    'mbpp/924': 'One conditional expression covers it.',
    'mbpp/970': 'A single comparison does it.',
    'mbpp/798': 'The built-in already does this.',
    'mbpp/813': '`len` is the whole job.',
    'mbpp/835': 'Rise over run, with a guard for vertical lines where the slope is undefined.',
    'mbpp/848': 'Average of the parallel sides times the height.',
    'mbpp/716': "A rhombus has four equal sides, so it's four times one side.",
    'mbpp/876': 'LCM falls out of the GCD: a·b // gcd(a, b).',
    'mbpp/672': '`max` takes any number of arguments.',
    'mbpp/688': "That's the magnitude, √(a² + b²).",
    'mbpp/877': 'Sort the characters and join them back.',
    'mbpp/518': '`math.isqrt` gives the exact integer root with no float error.',
    'mbpp/566': 'Peel digits off with `% 10` and `// 10`.',
    'mbpp/3': 'Trial division up to √n, skipping even divisors after 2.',
    'mbpp/935': 'Straight sum of squares over 1..n.',
    'mbpp/931': 'The sum of cubes is the square of the nth triangular number, so this is O(1).',
}

SKIP = {
    "mbpp/746": "expected values hard-code pi = 22/7; teaching that would be noise",
}

REPAIRS += [
    dict(id="mbpp/958",
         fault="The table has no subtractive pairs, so 4 comes out as 'IIII' instead of 'IV'.",
         pivot="Add CM, CD, XC, XL, IX and IV to the table; the greedy loop handles the rest.",
         code='def int_to_roman(num: int) -> str:\n    values = [\n        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),\n'
              '        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),\n        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),\n'
              '    ]\n    result = []\n    for value, symbol in values:\n        while num >= value:\n            result.append(symbol)\n'
              '            num -= value\n    return "".join(result)\n'),
    dict(id="mbpp/529",
         fault="The loop uses the Fibonacci step `a + b`; Jacobsthal-Lucas numbers follow L(n) = L(n−1) + 2·L(n−2).",
         pivot="Double the older term in the update.",
         edits=[["a, b = b, a + b", "a, b = b, b + 2 * a"]]),
    dict(id="mbpp/830",
         fault="`round` goes to the nearest value, not up, and the `+ 1` patch only works on whole numbers: 123.01247 "
               "becomes 123.0 at 0 digits.",
         pivot="Scale by 10**digits, take `math.ceil`, and scale back.",
         code="import math\n\n\ndef round_up(x, digits):\n    factor = 10 ** digits\n    return round(math.ceil(x * factor) / factor, digits)\n"),
    dict(id="mbpp/955",
         fault="`i = 1` also adds `n // 1 = n` to the divisors, so every number counts itself and 13 looks abundant.",
         pivot="Subtract n from the sum so only proper divisors count.",
         edits=[["proper_divisor_sum = sum(divisors)", "proper_divisor_sum = sum(divisors) - n"]]),
    dict(id="mbpp/738",
         fault="The loop multiplies by 2 each step, computing 2ⁿ⁻¹; the series is 1 + 1/2 + 1/4 + … + 1/2ⁿ, which "
               "approaches 2 (1.9921875 for n = 7).",
         pivot="Sum 1 / 2**i for i from 0 to n.",
         code="def geometric_sum(n):\n    return sum(1 / 2 ** i for i in range(n + 1))\n"),
    dict(id="mbpp/519",
         fault="The test passes one edge length, so this is a regular tetrahedron; the four-edge signature with no body "
               "can't even be called.",
         pivot="Take a single edge `a` and return a³ / (6√2), rounded to 2 places.",
         code="import math\n\n\ndef volume_tetrahedron(a):\n    return round(a ** 3 / (6 * math.sqrt(2)), 2)\n"),
    dict(id="mbpp/913",
         fault="`str.isdigit()` is called on the class with no argument, so it raises before anything is checked.",
         pivot="Test the last character directly; slicing with `s[-1:]` also keeps the empty string safe.",
         edits=[["return s.endswith(str.isdigit())", "return s[-1:].isdigit()"]]),
    dict(id="mbpp/568",
         fault="The list is hard-coded to five dicts and ignores `n`.",
         pivot="Build n distinct dicts with a comprehension (not `[{}] * n`, which would share one dict).",
         edits=[["return [{}, {}, {}, {}, {}]", "return [{} for _ in range(n)]"]]),
    dict(id="mbpp/799",
         fault="Rotating within `n.bit_length()` bits wraps 16 (0b10000) back onto itself; the expected 64 means a "
               "fixed 32-bit word.",
         pivot="Rotate inside a 32-bit width: `(n << d | n >> (32 - d)) & 0xFFFFFFFF`.",
         code="def left_Rotate(n, d, bits=32):\n    return ((n << d) | (n >> (bits - d))) & ((1 << bits) - 1)\n"),
    dict(id="mbpp/849",
         fault="Each prime is added once per multiplicity, so 60 = 2²·3·5 sums to 12; the task wants distinct prime "
               "divisors (2 + 3 + 5 = 10).",
         pivot="Add each prime once when it's found, then divide it out completely.",
         edits=[["        while temp % 2 == 0:\n            total += 2\n", "        total += 2\n        while temp % 2 == 0:\n"],
                ["            while temp % d == 0:\n                total += d\n",
                 "            total += d\n            while temp % d == 0:\n"]]),
    dict(id="mbpp/927",
         fault="The tests build trees with `Node(...)`, but no `Node` class exists, so setup fails before `max_height` runs.",
         pivot="Define a minimal `Node` with `data`, `left` and `right`; the height recursion itself is fine.",
         code="class Node:\n    def __init__(self, data):\n        self.data = data\n        self.left = None\n        self.right = None\n\n\n"
              "def max_height(root):\n    if root is None:\n        return 0\n\n    left_height = max_height(root.left)\n"
              "    right_height = max_height(root.right)\n\n    return 1 + max(left_height, right_height)\n"),
    dict(id="mbpp/629",
         fault="The test calls `Split`, but the function is named `split_even_numbers`.",
         pivot="Rename it to `Split`; the filter itself is right.",
         edits=[["def split_even_numbers(nums):", "def Split(nums):"]]),
    dict(id="mbpp/671",
         fault="Two problems: the test calls `set_Right_most_Unset_Bit`, and the loop condition `== 0` stops at the "
               "rightmost set bit, not the unset one.",
         pivot="Use the bit trick `n | (n + 1)`, returning n unchanged when it's already all ones (15 → 15).",
         code="def set_Right_most_Unset_Bit(n):\n    if n & (n + 1) == 0:  # already all ones\n        return n\n    return n | (n + 1)\n"),
    dict(id="mbpp/704",
         fault="`range(1, n)` stops at n − 1, so the 1/n term is missing (2.8289… instead of 2.9289… for n = 10).",
         pivot="Include n: `range(1, n + 1)`.",
         edits=[["for i in range(1, n):", "for i in range(1, n + 1):"]]),
]
SKIP.update({
    "mbpp/761": "expected values hard-code pi = 22/7",
    "mbpp/574": "expected values hard-code pi = 3.1415",
})

# cleaner first-pass code where the base model's passing draft has dead code or over-engineering
FIRST_CODE = {
    "mbpp/759": 'import re\n\n\ndef is_decimal(s):\n    return re.fullmatch(r"[0-9]+(\\.[0-9]{1,2})?", s) is not None\n',
    "mbpp/707": 'def count_Set_Bits(n):\n    return sum(bin(i).count("1") for i in range(1, n + 1))\n',
    "mbpp/520": "from math import gcd\n\n\ndef get_lcm(nums):\n    result = nums[0]\n    for n in nums[1:]:\n"
                "        result = result * n // gcd(result, n)\n    return result\n",
}

FRAMING.update({
    "mbpp/873": "Plain recursion as asked; fine for small n, though it's exponential.",
    "mbpp/605": "6k ± 1 trial division up to √n.",
    "mbpp/606": "Multiply by π/180.",
    "mbpp/654": "Twice the sum of the sides.",
    "mbpp/759": "A regex pins the shape exactly: digits, then optionally a dot and one or two decimals.",
    "mbpp/833": "Wrap the keys view in a list.",
    "mbpp/919": "A running product starting at 1.",
    "mbpp/707": "Counting bits number by number is O(n log n), fine at this size.",
    "mbpp/973": "Slicing does the rotation; `k % n` covers k past the length.",
    "mbpp/728": "`zip` pairs the elements and a comprehension adds them.",
    "mbpp/822": "Check each requirement with `any`, then the length.",
    "mbpp/903": "Unset bits per number are its bit length minus its set bits.",
    "mbpp/541": "Sum proper divisors in pairs up to √n and compare with n.",
    "mbpp/557": "`str.swapcase` does exactly this.",
    "mbpp/666": "`str.count` handles it.",
    "mbpp/685": "Sieve up to n, then sum what survives.",
    "mbpp/816": "Tuples are immutable, so clearing one means returning an empty tuple.",
    "mbpp/818": "Count the characters where `islower()` holds.",
    "mbpp/847": "A slice gives a shallow copy.",
    "mbpp/961": "Walk from the right and subtract any value smaller than the one after it.",
    "mbpp/678": "`split()` with no arguments drops all whitespace; join the pieces back.",
    "mbpp/789": "A regular polygon's perimeter is sides × length.",
    "mbpp/520": "Fold pairwise LCMs across the list using the GCD.",
    "mbpp/554": "Keep the elements with an odd remainder.",
    "mbpp/599": "Sum the range and divide by n.",
    "mbpp/742": "The surface area of a regular tetrahedron is √3 · a².",
    "mbpp/681": "Trial division: 2 first, then odd candidates up to √n; if nothing divides, n is prime.",
    "mbpp/852": "Filter out anything below zero.",
    "mbpp/855": "Even parity means an even count of set bits.",
    "mbpp/543": "Digit count of the sum via `len(str(...))`.",
    "mbpp/785": "Strip the parentheses, split on commas, convert each piece.",
})

REPAIRS += [
    dict(id="mbpp/853",
         fault="The test calls `sum_of_odd_Factors`, and stepping i by 2 misses odd factors whose partner is even "
               "(15 = 30 / 2 is never reached).",
         pivot="Divide out every factor of 2 first; the divisors of the odd part are exactly the odd factors.",
         code="def sum_of_odd_Factors(n):\n    while n % 2 == 0:\n        n //= 2\n"
              "    return sum(d for d in range(1, n + 1, 2) if n % d == 0)\n"),
    dict(id="mbpp/677",
         fault="These are angles, not side lengths: (60, 50, 90) passes the side inequality but sums to 200°, so it "
               "isn't a triangle.",
         pivot="Check that all three angles are positive and add up to 180.",
         code="def validity_triangle(a, b, c):\n    return a > 0 and b > 0 and c > 0 and a + b + c == 180\n"),
    dict(id="mbpp/783",
         fault="Saturation and value come back as fractions (1.0) while the test expects percentages (100.0).",
         pivot="Scale s and v by 100 on the way out; hue is already in degrees.",
         edits=[["return (h, s, v)", "return (h, s * 100, v * 100)"]]),
    dict(id="mbpp/837",
         fault="`n ** 4` isn't the sum of the first n odd cubes; 1³ + 3³ = 28, not 16.",
         pivot="Use the closed form n²(2n² − 1).",
         edits=[["return n ** 4", "return n * n * (2 * n * n - 1)"]]),
    dict(id="mbpp/851",
         fault="The call passes two arguments, n and the sum of its divisors; Σ 1/d over the divisors of n equals "
               "σ(n) / n, so the second argument already holds the answer.",
         pivot="Accept both and return `round(divisor_sum / n, 2)`.",
         code="def Sum_of_Inverse_Divisors(n, divisor_sum):\n    return round(divisor_sum / n, 2)\n"),
    dict(id="mbpp/768",
         fault="`n & -n` isolates the lowest set bit, so the count is always 1 and every nonzero number reports odd parity.",
         pivot="Count all the set bits of n itself.",
         code='def check_Odd_Parity(n):\n    return bin(n).count("1") % 2 == 1\n'),
    dict(id="mbpp/900",
         fault="No function was written, only an assert against an invented `starts_with_number`; the test calls "
               "`match_num` and expects True only for strings that begin with 5.",
         pivot="Define `match_num(s)` as a prefix check.",
         code='def match_num(s):\n    return s.startswith("5")\n'),
    dict(id="mbpp/511",
         fault="This sums all divisors, but the smallest sum of factors whose product is n comes from splitting n "
               "into primes: 12 = 2·2·3 gives 7.",
         pivot="Factor n and add the prime factors with multiplicity.",
         code="def find_Min_Sum(n):\n    total, d = 0, 2\n    while d * d <= n:\n        while n % d == 0:\n            total += d\n"
              "            n //= d\n        d += 1\n    return total + (n if n > 1 else 0)\n"),
    dict(id="mbpp/581",
         fault="The second argument is already the slant height (3² + 2·3·4 = 33), and each face is ½·base·slant, "
               "so the lateral area is 2·b·s, not 4·b·s on a recomputed slant.",
         pivot="Use b² + 2·b·s directly.",
         code="def surface_Area(base_side, slant_height):\n    return base_side ** 2 + 2 * base_side * slant_height\n"),
    dict(id="mbpp/624",
         fault="It compares against the literal 'PERSON' and returns a bool; the test wants the uppercased string.",
         pivot="Return `s.upper()`.",
         edits=[["def is_upper(s: str) -> bool:", "def is_upper(s: str) -> str:"],
                ['return s.upper() == "PERSON"', "return s.upper()"]]),
    dict(id="mbpp/957",
         fault="The loop clears one set bit per pass, so `pos` ends up as the popcount (2 for 12), not the position of "
               "the lowest set bit.",
         pivot="Isolate the lowest set bit with `n & -n` and take its `bit_length()` (12 → 0b100 → 3).",
         code="def get_First_Set_Bit_Pos(n: int) -> int:\n    return (n & -n).bit_length()\n"),
    dict(id="mbpp/521",
         fault="The task asks whether the triangle is scalene (despite the function name); `a == b or ...` tests for "
               "isosceles, the opposite.",
         pivot="Return True only when all three sides differ.",
         edits=[["return a == b or b == c or a == c", "return a != b and b != c and a != c"]]),
    dict(id="mbpp/717",
         fault="This is the population formula (÷ n); the expected 2.236 for [4, 2, 5, 8, 6] is the sample standard "
               "deviation (÷ n − 1).",
         pivot="Divide the squared deviations by n − 1.",
         edits=[["variance = sum((x - mean) ** 2 for x in data) / n", "variance = sum((x - mean) ** 2 for x in data) / (n - 1)"]]),
    dict(id="mbpp/592",
         fault="The test calls `sum_Of_product`, and the sum is of adjacent products C(n, k)·C(n, k+1), not squares: "
               "squares give 20 for n = 3 where 15 is expected.",
         pivot="Rename and multiply neighbouring coefficients.",
         code="from math import comb\n\n\ndef sum_Of_product(n):\n    return sum(comb(n, k) * comb(n, k + 1) for k in range(n))\n"),
    dict(id="mbpp/782",
         fault="The test calls `Odd_Length_Sum`, but the function is `odd_length_sum`.",
         pivot="Rename it; the enumeration already sums exactly the odd-length subarrays.",
         edits=[["def odd_length_sum(arr):", "def Odd_Length_Sum(arr):"]]),
    dict(id="mbpp/603",
         fault="Ludic numbers aren't primes: they come from a sieve that removes every k-th remaining number, and "
               "they include 1 and composites like 25.",
         pivot="Run the ludic sieve: take the first survivor k, then delete every k-th element starting with it.",
         code="def get_ludic(n):\n    ludics, rest = [1], list(range(2, n + 1))\n    while rest:\n        k = rest[0]\n"
              "        ludics.append(k)\n        del rest[::k]\n    return ludics\n"),
]
SKIP.update({"mbpp/882": "test expects 2*a*b for a 'perimeter'; spec is wrong"})

REPAIRS += [
    dict(id="mbpp/950",
         fault="`year % 12` puts the cycle's origin at year 0, but Rat years are 4, 16, …, 2020, so every sign is "
               "shifted and 1997 lands on Snake.",
         pivot="Offset the year by 4 before taking the remainder.",
         edits=[["pos = year % 12", "pos = (year - 4) % 12"]]),
    dict(id="mbpp/655",
         fault="The closed form is misremembered: Σ i⁵ = n²(n+1)²(2n² + 2n − 1) / 12, and the cubic factor here "
               "gives 54 for n = 2 instead of 33.",
         pivot="Fix the last factor to 2n² + 2n − 1.",
         edits=[["(2 * n**3 + n**2 - n)", "(2 * n * n + 2 * n - 1)"]]),
    dict(id="mbpp/963",
         fault="The math is right but the labels don't match: the test wants 'no real solution' and 'one solution', "
               "lower-case and singular.",
         pivot="Use the exact labels the test expects.",
         edits=[['("One solution", delta)', '("one solution", delta)'],
                ['("No real solutions", delta)', '("no real solution", delta)']]),
    dict(id="mbpp/587",
         fault="The test calls `list_tuple`, but the function is `list_to_tuple`.",
         pivot="Rename it; `tuple(lst)` is already the whole job.",
         edits=[["def list_to_tuple(lst):", "def list_tuple(lst):"]]),
    dict(id="mbpp/667",
         fault="The test passes the vowel set as a second argument, but the function only accepts the string.",
         pivot="Take `vowels` as a parameter and count membership in it.",
         code="def Check_Vow(s, vowels):\n    return sum(1 for ch in s if ch in vowels)\n"),
    dict(id="mbpp/558",
         fault="It counts positions where digits differ; the distance is the sum of absolute digit differences "
               "(|2−5| + |3−6| = 6).",
         pivot="Sum `abs(x - y)` over the paired digits.",
         edits=[["return sum(1 for x, y in zip(digits(a), digits(b)) if x != y)",
                 "return sum(abs(x - y) for x, y in zip(digits(a), digits(b)))"]]),
    dict(id="mbpp/553",
         fault="`t[0] + t[1]` adds the integers (4 + 56 = 60); the tuple holds the whole and fractional digits of one number.",
         pivot="Join them as text around a decimal point and parse that.",
         edits=[["return float(t[0] + t[1])", 'return float(f"{t[0]}.{t[1]}")']]),
    dict(id="mbpp/556",
         fault="The second argument is the array length, not an XOR target; a pair's XOR is odd exactly when one "
               "element is odd and the other even.",
         pivot="Count the odd elements and multiply by the even count.",
         code="def find_Odd_Pair(nums, n):\n    odd = sum(x & 1 for x in nums[:n])\n    return odd * (n - odd)\n"),
    dict(id="mbpp/808",
         fault="The input is a flat tuple of numbers, so each `t` is an int and `k in t` fails.",
         pivot="Test membership on the tuple itself.",
         code="def check_K(tup, k):\n    return k in tup\n"),
    dict(id="mbpp/831",
         fault="The test passes the array length as a second argument that the function doesn't accept.",
         pivot="Accept `n`; the Counter-based pair count is already right.",
         edits=[["def count_Pairs(arr):", "def count_Pairs(arr, n):"]]),
    dict(id="mbpp/564",
         fault="The test passes the array length as a second argument that the function doesn't accept.",
         pivot="Accept `n`; the pair loop itself is correct.",
         edits=[["def count_Pairs(arr):", "def count_Pairs(arr, n):"]]),
    dict(id="mbpp/755",
         fault="Raising on fewer than two distinct values breaks the expected contract: `[2, 2]` should return None.",
         pivot="Return None for both degenerate cases instead of raising.",
         edits=[['raise ValueError("List must contain at least two elements.")', "return None"],
                ['raise ValueError("List must contain at least two distinct numbers.")', "return None"]]),
]

FIRST_CODE.update({
    "mbpp/739": "def find_Index(n):\n    k = 1\n    while k * (k + 1) // 2 < 10 ** (n - 1):\n        k += 1\n    return k\n",
})

FRAMING.update({
    "mbpp/657": "Compute n! exactly, then drop digits from the right until one is left.",
    "mbpp/824": "Keep the odd ones.",
    "mbpp/909": "Step down from n − 1 until the digits read the same reversed.",
    "mbpp/929": "`tuple.count` does it.",
    "mbpp/598": "Raise each digit to the digit count, sum, and compare.",
    "mbpp/764": "Count the characters where `isdigit()` holds.",
    "mbpp/901": "That's the LCM of 1..n, folded with gcd.",
    "mbpp/904": "A remainder check.",
    "mbpp/905": "Sum the squared binomials (the total equals C(2n, n)).",
    "mbpp/646": "Each axis has n − k + 1 positions, so cube that.",
    "mbpp/937": "Count with a dict and take the key with the highest count.",
    "mbpp/947": "Minimum length over the words.",
    "mbpp/729": "`map` with a two-argument lambda walks both lists together.",
    "mbpp/565": "`list` on a string splits it into characters.",
    "mbpp/604": "Split on whitespace, reverse, join.",
    "mbpp/708": "`split()` breaks it on whitespace.",
    "mbpp/807": "Return the first number with an odd remainder.",
    "mbpp/945": "`set` takes the tuple directly.",
    "mbpp/959": "Sum over length.",
    "mbpp/739": "Walk k upward until the kth triangular number reaches n digits.",
})

def _rename(old, new, what="the logic itself is right"):
    return dict(fault=f"The test calls `{new}`, but the function is defined as `{old}`.",
                pivot=f"Rename it to `{new}`; {what}.", edits=[[f"def {old}(", f"def {new}("]])


REPAIRS += [
    dict(id="mbpp/793",
         fault="The third positional argument lands in `low`, so `last([1, 2, 3], 1, 3)` starts past the end; it's "
               "actually a length, and the arrays aren't reliably sorted ([2, 3, 2, 3, …]).",
         pivot="Scan from the right and return the first index that matches.",
         code="def last(arr, x, n):\n    for i in range(len(arr) - 1, -1, -1):\n        if arr[i] == x:\n            return i\n    return -1\n"),
    dict(id="mbpp/907",
         fault="Dropping multiples of 3 isn't the lucky-number rule; lucky numbers come from sieving the odd numbers, "
               "repeatedly deleting every k-th survivor where k is the next survivor (3, then 7, then 9, …).",
         pivot="Run that sieve over a range of odd numbers and return the first n.",
         code="def lucky_num(n):\n    nums = list(range(1, n * n + 10, 2))\n    i = 1\n"
              "    while i < len(nums) and nums[i] <= len(nums):\n        del nums[nums[i] - 1::nums[i]]\n        i += 1\n"
              "    return nums[:n]\n"),
    dict(id="mbpp/967",
         fault="Requiring all ten upper- and lower-case vowels rejects 'SEEquoiaL', and the failure label should be "
               "'not accepted'.",
         pivot="Compare the five vowels against `s.lower()` and use the expected labels.",
         edits=[['vowels = set("aeiouAEIOU")', 'vowels = set("aeiou")'],
                ['return "accepted" if vowels.issubset(set(s)) else "rejected"',
                 'return "accepted" if vowels <= set(s.lower()) else "not accepted"']]),
    dict(id="mbpp/668",
         fault="The test calls `replace`, and `s.replace(char, char, 1)` is a no-op anyway; the task is collapsing "
               "runs of the character into one.",
         pivot="Match two or more in a row with a regex and substitute a single copy.",
         code='import re\n\n\ndef replace(s, char):\n    return re.sub(re.escape(char) + "{2,}", char, s)\n'),
    dict(id="mbpp/531",
         fault="The call passes (coins, count, value), and greedy isn't safe for arbitrary coins anyway: for 11 with "
               "[9, 6, 5, 1] it takes 9 + 1 + 1 instead of 6 + 5.",
         pivot="Accept the count and use the minimum-coins DP over values 0..V.",
         code='def min_coins(coins, m, value):\n    INF = float("inf")\n    dp = [0] + [INF] * value\n'
              "    for v in range(1, value + 1):\n        for c in coins[:m]:\n            if c <= v and dp[v - c] + 1 < dp[v]:\n"
              "                dp[v] = dp[v - c] + 1\n    return dp[value]\n"),
    dict(id="mbpp/545",
         fault="Toggling bit 31 assumes a 32-bit word; the first bit here is the number's own highest bit "
               "(10 = 0b1010 → 0b0011 = 3).",
         pivot="Toggle `1 << (n.bit_length() - 1)` instead.",
         edits=[["n = n ^ (1 << 31)  # or 1 << 63 for 64-bit signed integers", "n = n ^ (1 << (n.bit_length() - 1))"]]),
    dict(id="mbpp/781", **_rename("count_divisors", "count_Divisors", "the sqrt-pair divisor count is right")),
    dict(id="mbpp/750",
         fault="A list can't be concatenated with a tuple directly.",
         pivot="Convert the tuple first: `lst + list(tup)`.",
         edits=[["return lst + tup", "return lst + list(tup)"]]),
    dict(id="mbpp/812",
         fault="The pattern is lower-case `road`, so it never matches 'Road', and the expected abbreviation is "
               "capitalised 'Rd.'.",
         pivot="Match `Road` and substitute `Rd.`.",
         edits=[['return re.sub(r"\\broad\\b", "rd.", s)', 'return re.sub(r"\\bRoad\\b", "Rd.", s)']]),
    dict(id="mbpp/964", **_rename("word_len_even", "word_len", "the parity check is right")),
    dict(id="mbpp/589",
         fault="`int(start ** 0.5)` rounds down, so when start isn't a square the first candidate falls below it "
               "(49 for start = 50).",
         pivot="Start from the smallest n with n² ≥ start: `math.isqrt(start - 1) + 1`.",
         edits=[["def perfect_squares(start, end):", "import math\n\n\ndef perfect_squares(start, end):"],
                ["n = int(start**0.5)", "n = math.isqrt(start - 1) + 1"]]),
    dict(id="mbpp/625", **_rename("swap_list", "swap_List", "the swap is right")),
    dict(id="mbpp/684",
         fault="Beyond the name (`count_Char`), the count is over the string repeated out to 10 characters: 'aba' "
               "gives 7, not 2.",
         pivot="Count whole repeats times `s.count(c)`, plus the partial prefix.",
         code="def count_Char(s, c, length=10):\n    full, rem = divmod(length, len(s))\n    return s.count(c) * full + s[:rem].count(c)\n"),
    dict(id="mbpp/841",
         fault="The test passes the length as a second argument, and the 'merge' loop isn't a real merge: it appends "
               "to an already sorted copy and counts across unrelated ends.",
         pivot="Count pairs i < j with arr[i] > arr[j] directly; n is small.",
         code="def get_inv_count(arr, n):\n    return sum(1 for i in range(n) for j in range(i + 1, n) if arr[i] > arr[j])\n"),
    dict(id="mbpp/517", **_rename("largest_positive", "largest_pos", "the scan is right")),
    dict(id="mbpp/536",
         fault="`lst[:n]` takes the first n items; every nth item is a stride, `[1, 3, 5, 7, 9]` for n = 2.",
         pivot="Slice with a step: `lst[::n]`.",
         code="def nth_items(lst, n):\n    return lst[::n]\n"),
    dict(id="mbpp/744", **_rename("has_none", "check_none", "`None in t` is right")),
]

FRAMING.update({
    "mbpp/845": "Compute n! exactly and count its digits.",
    "mbpp/664": "Average of the even numbers from 2 to n.",
    "mbpp/665": "Slice off the head and append it.",
    "mbpp/731": "Lateral area is π·r·l, with slant height l = √(r² + h²).",
    "mbpp/771": "A stack of open brackets; every closer has to match the top.",
    "mbpp/549": "Sum (2k − 1)⁵ for k up to n.",
    "mbpp/562": "Track the longest length seen.",
    "mbpp/577": "Keep the product mod 10 as it grows.",
    "mbpp/682": "Pairwise product over `zip`.",
    "mbpp/692": "Keep the product mod 100 as it grows.",
    "mbpp/770": "Sum (2k − 1)⁴ for k up to n.",
    "mbpp/892": "Split on whitespace and rejoin with single spaces.",
    "mbpp/673": "Concatenate the digits as text and parse once.",
    "mbpp/618": "Pairwise division over `zip`.",
    "mbpp/637": "No profit and no loss means the two amounts are equal.",
})

REPAIRS += [
    dict(id="mbpp/948",
         fault="The guard rejects negative indices, but `tup[-4]` is valid Python and the test expects 'u'.",
         pivot="Drop the guard and index directly; a truly out-of-range index still raises on its own.",
         edits=[['    if index < 0 or index >= len(tup):\n        raise IndexError("index out of range")\n', ""]]),
    dict(id="mbpp/968",
         fault="The periodic function is floor(A·x / B) − A·floor(x / B), not floor((n + x) / k); it peaks at "
               "x = min(B − 1, N), where the second term is still 0.",
         pivot="Evaluate A·x // B at that x.",
         code="def floor_Max(A, B, N):\n    x = min(B - 1, N)\n    return A * x // B\n"),
    dict(id="mbpp/515",
         fault="The call is (nums, n, m), so the divisor 6 never reaches `m`, and `prefix_sums[i]` mixes indices with "
               "remainders (KeyError: 1); prefix sums only cover contiguous runs, not subsets.",
         pivot="Track the set of reachable subset-sum remainders mod m and stop when 0 appears.",
         code="def modular_sum(nums, n, m):\n    seen = set()\n    for x in nums[:n]:\n"
              "        seen |= {(r + x) % m for r in seen} | {x % m}\n        if 0 in seen:\n            return True\n    return False\n"),
    dict(id="mbpp/640",
         fault="The input is a one-element list, not a string, so the slicing returns the list untouched; the loop "
               "also only trims the end and never looks for the opening parenthesis.",
         pivot="Pull the string out of the list and delete `(…)` plus the space before it with one regex.",
         code='import re\n\n\ndef remove_parenthesis(items):\n    return re.sub(r"\\s*\\([^)]*\\)", "", items[0])\n'),
    dict(id="mbpp/701",
         fault="`arr[i]` joins the left sum before the comparison, so the pivot is counted on one side; an "
               "equilibrium index excludes itself from both.",
         pivot="Compare after removing arr[i] from the right but before adding it to the left.",
         edits=[["        left_sum += arr[i]\n        right_sum -= arr[i]\n        if left_sum == right_sum:\n            return i\n",
                 "        right_sum -= arr[i]\n        if left_sum == right_sum:\n            return i\n        left_sum += arr[i]\n"]]),
    dict(id="mbpp/734",
         fault="The test calls `sum_Of_Subarray_Prod(arr, n)`, and dividing prefix products breaks as soon as the "
               "array contains a 0.",
         pivot="Rename, accept n, and use the running recurrence: products ending at i = arr[i] · (products ending at i−1 + 1).",
         code="def sum_Of_Subarray_Prod(arr, n):\n    total = ending_here = 0\n    for x in arr[:n]:\n"
              "        ending_here = x * (ending_here + 1)\n        total += ending_here\n    return total\n"),
    dict(id="mbpp/820",
         fault="It returns True for every month; only February (2) has 28 days.",
         pivot="Return `month == 2`.",
         edits=[["    return True", "    return month == 2"]]),
    dict(id="mbpp/593",
         fault="The tests include a three-part address ('12.01.024'), so the strict four-octet check raises before "
               "anything is stripped.",
         pivot="Drop the format check; `str(int(part))` already strips the zeros.",
         edits=[['    if len(parts) != 4:\n        raise ValueError("Invalid IPv4 address format")\n', ""]]),
    dict(id="mbpp/651",
         fault="The arguments are reversed: the call passes the larger tuple first and asks whether the second is "
               "contained in it.",
         pivot="Swap the parameter order; the `all(... in ...)` check is right.",
         edits=[["def check_subset(subset, superset):", "def check_subset(superset, subset):"]]),
    dict(id="mbpp/663",
         fault="The arguments are (n, x, y) with the bound first: find the largest k ≤ n with k % x == y; "
               "`x * y + y` ignores the bound entirely.",
         pivot="Step down from n by `(n - y) % x` to land on the residue.",
         code="def find_max_val(n, x, y):\n    k = n - (n - y) % x\n    return k if k >= 0 else -1\n"),
    dict(id="mbpp/880",
         fault="The no-root label should be 'No solutions', not '0 solutions'.",
         pivot="Match the expected string.",
         edits=[['return "0 solutions"', 'return "No solutions"']]),
    dict(id="mbpp/881",
         fault="It adds every even and every odd number; the task wants only the first even plus the first odd (4 + 1 = 5).",
         pivot="Take the first of each with `next(...)`.",
         edits=[["return sum(x for x in lst if x % 2 == 0) + sum(x for x in lst if x % 2 != 0)",
                 "return next(x for x in lst if x % 2 == 0) + next(x for x in lst if x % 2 != 0)"]]),
    dict(id="mbpp/953",
         fault="The test calls `subset(arr, n)`, and the count of distinct values isn't the answer: each subset holds "
               "a value at most once, so you need as many subsets as the most frequent value appears.",
         pivot="Return the maximum frequency.",
         code="from collections import Counter\n\n\ndef subset(arr, n):\n    return max(Counter(arr[:n]).values())\n"),
    dict(id="mbpp/632",
         fault="Besides the name (`move_zero`), the padding appends `len(nums)` zeros instead of only the ones removed.",
         pivot="Rename and pad with `len(nums) - len(non_zero)` zeros.",
         edits=[["def move_zeroes(nums):", "def move_zero(nums):"],
                ["return non_zero + [0] * len(nums)", "return non_zero + [0] * (len(nums) - len(non_zero))"]]),
    dict(id="mbpp/804",
         fault="The test passes the length as a second argument that the function doesn't accept.",
         pivot="Accept `n`; 'any even factor' is already the right test.",
         edits=[["def is_Product_Even(numbers):", "def is_Product_Even(numbers, n):"]]),
    dict(id="mbpp/940",
         fault="The sort works in place, but `heap_sort` returns nothing, so the caller gets None.",
         pivot="Return `arr` at the end.",
         edits=[["        heapify(arr, i, 0)  # restore heap property for the remaining elements\n",
                 "        heapify(arr, i, 0)  # restore heap property for the remaining elements\n    return arr\n"]]),
    dict(id="mbpp/626",
         fault="The test passes a single radius; the largest triangle in a semicircle uses the diameter as base and "
               "the radius as height, so its area is r², with -1 for a negative radius.",
         pivot="Take `r` and return `r * r`, guarding negatives.",
         code="def triangle_area(r):\n    if r < 0:\n        return -1\n    return r * r\n"),
    dict(id="mbpp/735",
         fault="The draft never returns, and its mask is built from n's value (`1 << (n - 2)`) instead of its bit length.",
         pivot="Mask the bits strictly between the top bit and bit 0, `(1 << (bit_length - 1)) - 2`, and XOR.",
         code="def toggle_middle_bits(n):\n    if n < 4:\n        return n\n"
              "    mask = (1 << (n.bit_length() - 1)) - 2  # ones strictly between the top bit and bit 0\n    return n ^ mask\n"),
]
SKIP.update({"mbpp/609": "'minimum' spec contradicts the expected values",
             "mbpp/714": "spec ('distinct power of prime factor') too ambiguous to diagnose from the prompt"})

FIRST_CODE.update({
    "mbpp/724": "def power_base_sum(base, power):\n    return sum(int(d) for d in str(base ** power))\n",
})
FRAMING.update({
    "mbpp/962": "Snap the bounds to even numbers and sum the evens between them.",
    "mbpp/634": "Sum (2k)⁴ for k up to n.",
    "mbpp/800": "Split on any whitespace and join with nothing.",
    "mbpp/887": "The low bit is 1 exactly when n is odd.",
    "mbpp/550": "Binary search for the peak where the rotation breaks the order.",
    "mbpp/724": "Python ints are exact, so compute the power and add up its digits.",
    "mbpp/866": "Membership in the list of 31-day months.",
    "mbpp/941": "Count until the first tuple shows up.",
    "mbpp/567": "Every adjacent pair has to be in order.",
    "mbpp/762": "April, June, September and November are the 30-day months.",
    "mbpp/797": "Sum the odd numbers between l and r.",
})

REPAIRS += [
    dict(id="mbpp/741", **_rename("all_characters_same", "all_Characters_Same", "the comparison loop is right")),
    dict(id="mbpp/546",
         fault="Three mismatches with the test: the name is `last_occurence_char` (one r), the loop returns the first "
               "match rather than the last, and positions are 1-based with None when absent.",
         pivot="Use `s.rfind(c)`, add 1, and return None on -1.",
         code="def last_occurence_char(s, c):\n    i = s.rfind(c)\n    return i + 1 if i >= 0 else None\n"),
    dict(id="mbpp/602",
         fault="The test expects the string 'None' when nothing repeats, and counting first can pick the wrong "
               "character: in 'abba' the first repeat is 'b', but the Counter loop returns 'a'.",
         pivot="Scan once with a seen-set, return the first character already seen, else 'None'.",
         code='def first_repeated_char(s):\n    seen = set()\n    for ch in s:\n        if ch in seen:\n            return ch\n'
              '        seen.add(ch)\n    return "None"\n'),
    dict(id="mbpp/787",
         fault="Every branch returns 'Not matched!', and `startswith` would miss a match later in the string anyway.",
         pivot="Search for `ab{3}` anywhere and return the matching label.",
         code='import re\n\n\ndef text_match_three(text):\n    return "Found a match!" if re.search(r"ab{3}", text) else "Not matched!"\n'),
    dict(id="mbpp/801",
         fault="With no equal pair the count should be 0, not 1.",
         pivot="Return 0 when all three differ.",
         edits=[["    return 1\n", "    return 0\n"]]),
    dict(id="mbpp/767",
         fault="The test passes the length before the target, and the loop counts every unequal pair twice, once "
               "from each side.",
         pivot="Accept n, and count each unordered pair once by only taking x < target − x.",
         code="from collections import Counter\n\n\ndef get_Pairs_Count(nums, n, target):\n    count = Counter(nums[:n])\n"
              "    total = 0\n    for x in count:\n        y = target - x\n        if x == y:\n"
              "            total += count[x] * (count[x] - 1) // 2\n        elif x < y and y in count:\n"
              "            total += count[x] * count[y]\n    return total\n"),
    dict(id="mbpp/537",
         fault="Counter returns the first word that repeats anywhere ('ab'), not the word whose repeat comes first "
               "('ca'), and the no-repeat case should return the string 'None'.",
         pivot="Scan once with a seen-set and return on the first hit.",
         code='def first_repeated_word(s):\n    seen = set()\n    for word in s.split():\n        if word in seen:\n'
              '            return word\n        seen.add(word)\n    return "None"\n'),
    dict(id="mbpp/650",
         fault="The arrays count as equal when they hold the same elements in any order ([1, 2, 3] vs [3, 2, 1]), so "
               "an element-by-element comparison is too strict.",
         pivot="Compare the lengths, then the sorted contents.",
         code="def are_Equal(arr1, arr2, n, m):\n    return n == m and sorted(arr1) == sorted(arr2)\n"),
    dict(id="mbpp/751",
         fault="The test calls `check_min_heap(arr, 0)`: a different name plus a start index.",
         pivot="Rename and accept the index; the parent-child checks are right.",
         edits=[["def is_min_heap(arr):", "def check_min_heap(arr, i=0):"]]),
    dict(id="mbpp/777",
         fault="'Non-repeated' here means each distinct value once (the expected 21 includes the 1 that appears three "
               "times), and the second argument is the length, not a cap.",
         pivot="Sum the distinct values: `sum(set(arr[:k]))`.",
         edits=[["    return sum(non_repeated[:k])", "    return sum(set(arr[:k]))"]]),
    dict(id="mbpp/856",
         fault="The test passes the length as a second argument, and `list.count` takes no start index, so the "
               "zeros-to-the-right count can't work as written.",
         pivot="Walk from the right counting zeros, and add that count at each 1.",
         code="def find_Min_Swaps(arr, n):\n    swaps = zeros_right = 0\n    for x in reversed(arr[:n]):\n        if x == 0:\n"
              "            zeros_right += 1\n        else:\n            swaps += zeros_right\n    return swaps\n"),
    dict(id="mbpp/516",
         fault="The inner loop reassigns `digit`, the pass counter, so after the first element every number is "
               "bucketed by the wrong place value.",
         pivot="Give the per-number digit its own name.",
         edits=[["            digit = (num // (10 ** digit)) % 10\n            buckets[digit].append(num)",
                 "            d = (num // (10 ** digit)) % 10\n            buckets[d].append(num)"]]),
    dict(id="mbpp/719",
         fault="'a followed by zero or more b's' is a pattern to find, not a rule for the whole string: 'ac' matches, "
               "and the miss label should be 'Not matched!'.",
         pivot="Search for `ab*` and return the expected labels.",
         code='import re\n\n\ndef text_match(text):\n    return "Found a match!" if re.search(r"ab*", text) else "Not matched!"\n'),
    dict(id="mbpp/965",
         fault="The function falls off the end without returning and never lower-cases; the first substitution was "
               "already enough.",
         pivot="Keep the first regex, lower-case the result, and return it.",
         code='import re\n\n\ndef camel_to_snake(s):\n    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()\n'),
    dict(id="mbpp/791",
         fault="It recurses into nested tuples and keeps them; the task is to drop them.",
         pivot="Filter out tuple items instead of descending into them.",
         code="def remove_nested(t):\n    return tuple(item for item in t if not isinstance(item, tuple))\n"),
    dict(id="mbpp/893",
         fault="The comprehension flattens every element; only each sublist's last item is wanted.",
         pivot="Take `sub[-1]` per sublist.",
         edits=[["return [x for sublist in lst for x in sublist if x]", "return [sublist[-1] for sublist in lst]"]]),
    dict(id="mbpp/954",
         fault="The second argument is the cost, not the profit; profit is the difference, 1500 − 1200 = 300.",
         pivot="Return `amount - cost` when it's positive, else None.",
         code="def profit_amount(amount, cost):\n    return amount - cost if amount > cost else None\n"),
    dict(id="mbpp/5",
         fault="That recurrence is for a 2 × n board; on a 3 × n board odd widths have no tilings and even widths "
               "follow A(n) = 4·A(n−2) − A(n−4), giving 153 for n = 8.",
         pivot="Return 0 for odd n and iterate the 3 × n recurrence from A(0) = 1, A(2) = 3.",
         code="def count_ways(n):\n    if n % 2:\n        return 0\n    if n == 0:\n        return 1\n"
              "    a, b = 1, 3  # widths 0 and 2\n    for _ in range(n // 2 - 1):\n        a, b = b, 4 * b - a\n    return b\n"),
    dict(id="mbpp/620",
         fault="No code came out: the reply argued in circles about GCDs. Sorted, a pairwise-divisible subset is a "
               "chain where each element divides the next, so this is a longest-chain DP.",
         pivot="Sort, then for each element extend the best chain ending at any earlier divisor.",
         code="def largest_subset(a, n):\n    a = sorted(a[:n])\n    best = [1] * n\n    for i in range(n):\n"
              "        for j in range(i):\n            if a[i] % a[j] == 0:\n                best[i] = max(best[i], best[j] + 1)\n"
              "    return max(best)\n"),
]
SKIP.update({"mbpp/535": "expected values hard-code pi = 3.1415"})

FIRST_CODE.update({
    "mbpp/784": "def mul_even_odd(lst):\n    first_even = next(x for x in lst if x % 2 == 0)\n"
                "    first_odd = next(x for x in lst if x % 2 != 0)\n    return first_even * first_odd\n",
    "mbpp/594": "def diff_even_odd(lst):\n    first_even = next(x for x in lst if x % 2 == 0)\n"
                "    first_odd = next(x for x in lst if x % 2 != 0)\n    return first_even - first_odd\n",
})
FRAMING.update({
    "mbpp/687": "Euclid's algorithm, recursively.",
    "mbpp/796": "Sum the dictionary's values.",
    "mbpp/803": "`math.isqrt` avoids float error on large n.",
    "mbpp/600": "Even numbers have a 0 low bit.",
    "mbpp/784": "Grab the first even and first odd with `next` and multiply.",
    "mbpp/525": "Lines are parallel when their (a, b) coefficients are proportional, i.e. the cross product is 0.",
    "mbpp/627": "Return the first value in the range that isn't in the set.",
    "mbpp/715": "Split on commas and convert each piece.",
    "mbpp/736": "Binary search for the first index whose value isn't below x.",
    "mbpp/594": "First even minus first odd, each found with `next`.",
})

REPAIRS += [
    dict(id="mbpp/633",
         fault="Two operator slips: the expected 47 is the sum of pairwise XORs, but the loop ORs each pair and ORs "
               "that into the total, which saturates at 15.",
         pivot="Accumulate `total += arr[i] ^ arr[j]`.",
         edits=[["total |= arr[i] | arr[j]", "total += arr[i] ^ arr[j]"]]),
    dict(id="mbpp/878",
         fault="`k` is a list of allowed values, not a count; the check is whether every element of the tuple is one of them.",
         pivot="Compare as sets: `set(t) <= set(k)`.",
         edits=[["return len(t) == k", "return set(t) <= set(k)"]]),
    dict(id="mbpp/514",
         fault="The input is a flat tuple of numbers, not a list of tuples, so the inner loop tries to iterate an int.",
         pivot="Sum it directly.",
         edits=[["return sum(elem for t in tuples for elem in t)", "return sum(tuples)"]]),
    dict(id="mbpp/868", **_rename("length_of_last_word", "length_Of_Last_Word", "split-and-measure is right")),
    dict(id="mbpp/908",
         fault="A fixed point is an index i with arr[i] == i; the heap returns the k-th smallest value, which is unrelated.",
         pivot="Scan for the first index whose value equals it, else -1.",
         code="def find_fixed_point(arr, n):\n    for i, x in enumerate(arr[:n]):\n        if x == i:\n            return i\n    return -1\n"),
    dict(id="mbpp/974",
         fault="Row 0 has a single entry but the base case loops over all n columns, and `dp[i - 1][j - 1]` wraps to "
               "the last column when j = 0.",
         pivot="Go bottom-up instead: each cell adds the smaller of its two children, and the top ends up holding the answer.",
         code="def min_sum_path(triangle):\n    best = list(triangle[-1])\n    for row in reversed(triangle[:-1]):\n"
              "        best = [x + min(best[j], best[j + 1]) for j, x in enumerate(row)]\n    return best[0]\n"),
    dict(id="mbpp/656",
         fault="The draft got lost in a DP that never finished (it ends inside a `for` with no body); pairing the two "
               "arrays in sorted order already minimises the total absolute difference.",
         pivot="Sort both and sum the paired differences.",
         code="def find_Min_Sum(a, b, n):\n    return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b)))\n"),
    dict(id="mbpp/836",
         fault="The second argument is the array length, not a window size, so the 'window' is the whole array; the "
               "task is the length of the maximum-sum subarray.",
         pivot="Run Kadane's algorithm and remember where the current run started.",
         code='def max_sub_array_sum(a, size):\n    best, cur = float("-inf"), 0\n    start = best_len = 0\n'
              "    for i, x in enumerate(a[:size]):\n        if cur <= 0:\n            cur, start = x, i\n        else:\n"
              "            cur += x\n        if cur > best:\n            best, best_len = cur, i - start + 1\n    return best_len\n"),
    dict(id="mbpp/552",
         fault="The test calls `Seq_Linear`, and the failure label should be 'Non Linear Sequence'.",
         pivot="Rename and fix the label; the constant-difference check is right.",
         code='def Seq_Linear(seq):\n    step = seq[1] - seq[0]\n    for i in range(1, len(seq)):\n'
              '        if seq[i] - seq[i - 1] != step:\n            return "Non Linear Sequence"\n    return "Linear Sequence"\n'),
    dict(id="mbpp/575",
         fault="It counts every non-multiple in the range, but the task wants the nth such number, and the first "
               "argument is the divisor: in (2, 3, 1, 10) that's the 3rd odd number from 1, i.e. 5.",
         pivot="Walk the range, count non-multiples of the first argument, and return the value at the nth.",
         code="def count_no(a, n, l, r):\n    count = 0\n    for i in range(l, r + 1):\n        if i % a != 0:\n"
              "            count += 1\n            if count == n:\n                return i\n    return -1\n"),
    dict(id="mbpp/802",
         fault="The test passes the length as a second argument, and the count is off: the array was rotated by the "
               "index after the drop (i + 1), not n − 1 − i.",
         pivot="Accept n and return i + 1 at the first descent.",
         edits=[["def count_Rotation(arr):", "def count_Rotation(arr, n):"], ["return n - 1 - i", "return i + 1"]]),
    dict(id="mbpp/867",
         fault="The second argument is the length, not a target sum; the task is the smallest positive number that "
               "makes the sum even: 1 if the sum is odd, otherwise 2.",
         pivot="Branch on the parity of the sum.",
         code="def min_Num(arr, n):\n    return 1 if sum(arr) % 2 else 2\n"),
    dict(id="mbpp/713",
         fault="`isinstance(x, bool)` is True for False as well; 'valid' means every value is truthy.",
         pivot="Return `all(t)`.",
         edits=[["return all(isinstance(x, bool) for x in t)", "return all(t)"]]),
    dict(id="mbpp/838",
         fault="Beyond the name (`min_Swaps`), mismatches come in two kinds, 0/1 and 1/0: two of the same kind fix "
               "with one swap but a mixed pair needs two, so '00011' vs '01001' takes 2, not 1.",
         pivot="Count each mismatch type: same-type pairs cost 1, a leftover mixed pair costs 2.",
         code='def min_Swaps(s1, s2):\n    c0 = sum(a == "0" and b == "1" for a, b in zip(s1, s2))\n'
              '    c1 = sum(a == "1" and b == "0" for a, b in zip(s1, s2))\n    if (c0 + c1) % 2:\n        return -1\n'
              "    return c0 // 2 + c1 // 2 + 2 * (c0 % 2)\n"),
    dict(id="mbpp/597",
         fault="The arguments are (arr1, arr2, m, n, k) with k last, and comparing k against each array's own index "
               "ignores how many elements the other array has already contributed.",
         pivot="Merge the two sorted arrays and take the (k−1)th element.",
         code="import heapq\n\n\ndef find_kth(arr1, arr2, m, n, k):\n    return list(heapq.merge(arr1[:m], arr2[:n]))[k - 1]\n"),
    dict(id="mbpp/756",
         fault="'a followed by zero or one b' is a pattern to search for: 'ac' contains it, and the miss label should "
               "be 'Not matched!'.",
         pivot="Search for `ab?` and return the expected labels.",
         code='import re\n\n\ndef text_match_zero_one(text):\n    return "Found a match!" if re.search(r"ab?", text) else "Not matched!"\n'),
    dict(id="mbpp/760",
         fault="Beyond the name (`unique_Element`) and the missing length argument, the loop answers 'YES' on the "
               "first duplicate, which isn't the same as having only one distinct value.",
         pivot="Return 'YES' exactly when the set of values has size 1.",
         code='def unique_Element(arr, n):\n    return "YES" if len(set(arr[:n])) == 1 else "NO"\n'),
    dict(id="mbpp/806",
         fault="`max_len` is never updated inside the loop, so only the final run counts, and every run starts at 1 "
               "even on lowercase letters.",
         pivot="Track the current run per character and fold it into the best as you go.",
         code="def max_run_uppercase(s):\n    best = run = 0\n    for ch in s:\n        run = run + 1 if ch.isupper() else 0\n"
              "        best = max(best, run)\n    return best\n"),
]

FIRST_CODE.update({
    "mbpp/576": "def is_Sub_Array(a, b, n, m):\n    return any(a[i:i + m] == b for i in range(n - m + 1))\n",
    "mbpp/828": "def count_alpha_dig_spl(s):\n    alpha = sum(ch.isalpha() for ch in s)\n    digit = sum(ch.isdigit() for ch in s)\n"
                "    return alpha, digit, len(s) - alpha - digit\n",
})
FRAMING.update({
    "mbpp/886": "Mean: total divided by count.",
    "mbpp/972": "Tuple `+` concatenates.",
    "mbpp/532": "Compare character counts with `Counter`.",
    "mbpp/710": "First and last by index.",
    "mbpp/786": "Binary search for the insertion point.",
    "mbpp/832": "Pull every digit run with `\\d+` and take the largest.",
    "mbpp/576": "Slide a window of b's length across a and compare.",
    "mbpp/865": "`map` with a lambda scales each element by n.",
    "mbpp/874": "s has to be t repeated a whole number of times.",
    "mbpp/891": "Compare how many digits each has.",
    "mbpp/928": "Split on '-' and reverse the order.",
    "mbpp/588": "max minus min.",
    "mbpp/9": "Try each rotation length and stop at the first one that reproduces the string.",
    "mbpp/726": "Multiply each element by its right neighbour.",
    "mbpp/737": "Anchor a vowel class at the start of the string.",
    "mbpp/828": "Count letters and digits; everything else is special.",
    "mbpp/850": "Positive angles that sum to 180.",
})

REPAIRS += [
    dict(id="mbpp/890",
         fault="The test passes the shorter array's length as a third argument, and when the extra element sits at the "
               "end its index is `len(arr2)`, not `len(arr1)`.",
         pivot="Accept n and return `len(arr2)` when the shared prefix matches completely.",
         edits=[["def find_Extra(arr1, arr2):", "def find_Extra(arr1, arr2, n):"],
                ["        return len(arr1)\n", "        return len(arr2)\n"]]),
    dict(id="mbpp/788",
         fault="The arguments are (list, string), not (string, list), and the string belongs in the tuple whole, not "
               "split into characters.",
         pivot="Return `tuple(lst + [s])`.",
         code="def new_tuple(lst, s):\n    return tuple(lst + [s])\n"),
    dict(id="mbpp/669",
         fault="The comprehension reads `m.group(i)`, but the match object is named `match`.",
         pivot="Use the right name; the octet range check is fine.",
         edits=[["octets = [int(m.group(i)) for i in range(1, 5)]", "octets = [int(match.group(i)) for i in range(1, 5)]"]]),
    dict(id="mbpp/871",
         fault="`s1 + s2 == s2 + s1` tests whether the strings commute, not whether one is a rotation of the other.",
         pivot="A rotation of s1 always appears inside s1 + s1, given equal lengths.",
         edits=[["return s1 + s2 == s2 + s1", "return len(s1) == len(s2) and s2 in s1 + s1"]]),
    dict(id="mbpp/528",
         fault="The test calls `min_length` and expects the minimum length together with the list itself, `(1, [0])`, "
               "not just the length.",
         pivot="Take `min(lists, key=len)` and return its length with it.",
         code="def min_length(lists):\n    shortest = min(lists, key=len)\n    return len(shortest), shortest\n"),
    dict(id="mbpp/706",
         fault="The call is (superset, m, subset, n): each array is followed by its length, so the parameters are "
               "misaligned from the second one on.",
         pivot="Take the four arguments in that order and compare as sets.",
         code="def is_subset(arr1, m, arr2, n):\n    return set(arr2[:n]) <= set(arr1[:m])\n"),
    dict(id="mbpp/942",
         fault="The first argument is a tuple of values, not a single target, so `target in candidates` looks for the "
               "whole tuple; the question is whether any of its elements appears in the list.",
         pivot="Check each element: `any(x in candidates for x in target)`.",
         edits=[["return target in candidates", "return any(x in candidates for x in target)"]]),
    dict(id="mbpp/547",
         fault="The test passes only n, and the distance is between consecutive numbers (i and i + 1), not from 0 to each i.",
         pivot="Sum the popcount of `i ^ (i + 1)` for i below n.",
         code='def Total_Hamming_Distance(n):\n    return sum(bin(i ^ (i + 1)).count("1") for i in range(n))\n'),
    dict(id="mbpp/572",
         fault="The input is one flat list, not a list of lists, so the inner loop iterates an int; and the result "
               "should keep every number that appears once, not just two of them.",
         pivot="Count the flat list and keep the values with count 1, in order.",
         code="from collections import Counter\n\n\ndef two_unique_nums(nums):\n    counts = Counter(nums)\n"
              "    return [x for x in nums if counts[x] == 1]\n"),
    dict(id="mbpp/591", **_rename("swap_list", "swap_List", "the swap is right")),
    dict(id="mbpp/610",
         fault="k is 1-based here: removing the 3rd element of [1, 1, 2, …] drops the 2, but `lst[:k] + lst[k+1:]` "
               "removes index 3.",
         pivot="Slice around index k − 1 and shift the bounds check to match.",
         edits=[["if k < 0 or k >= len(lst):", "if k < 1 or k > len(lst):"],
                ["return lst[:k] + lst[k+1:]", "return lst[:k - 1] + lst[k:]"]]),
    dict(id="mbpp/660",
         fault="The answer is a pair of points, the smaller left endpoint and the larger right endpoint, which can't "
               "lie in both ranges; the draft returns a single number.",
         pivot="Return `(min(l1, l2), max(r1, r2))`, with -1 where the endpoints coincide.",
         code="def find_Points(l1, r1, l2, r2):\n    x = min(l1, l2) if l1 != l2 else -1\n"
              "    y = max(r1, r2) if r1 != r2 else -1\n    return (x, y)\n"),
    dict(id="mbpp/679",
         fault="`d[index]` looks up the key `0`, not the 0th key; dicts aren't indexable by position.",
         pivot="Index the key list: `list(d)[index]`.",
         edits=[["return d[index]", "return list(d)[index]"]]),
    dict(id="mbpp/2",
         fault="The test expects a tuple in ascending order ((13, 14) even though t1 lists 14 first), and the "
               "comprehension returns a list in t1's order.",
         pivot="Intersect as sets and return a sorted tuple.",
         code="def similar_elements(t1, t2):\n    return tuple(sorted(set(t1) & set(t2)))\n"),
    dict(id="mbpp/611",
         fault="`max(t for t ...)` compares whole tuples, so it returns the largest row instead of the largest value "
               "in column n.",
         pivot="Take `t[n]` from each row before the max.",
         edits=[["return max(t for t in tuples if n < len(t))", "return max(t[n] for t in tuples)"]]),
]
SKIP.update({"mbpp/826": "expected labels follow a buggy reference (1, 0, 1 is 'right-angled')",
             "mbpp/670": "test expects increasing sequences to count as 'decreasing'"})

FIRST_CODE.update({
    "mbpp/683": "import math\n\n\ndef sum_Square(n):\n    a = 0\n    while 2 * a * a <= n:\n        b = math.isqrt(n - a * a)\n"
                "        if a * a + b * b == n:\n            return True\n        a += 1\n    return False\n",
    "mbpp/647": 'import re\n\n\ndef split_upperstring(s):\n    return re.findall(r"[A-Z][^A-Z]*", s)\n',
    "mbpp/811": "def check_identical(list1, list2):\n    return list1 == list2\n",
})
FRAMING.update({
    "mbpp/675": "Compute the sum and swap in 20 when it lands inside the range.",
    "mbpp/790": "Step through the even indices and check each value's parity.",
    "mbpp/683": "Try each a up to √(n/2) and check whether n − a² is a perfect square.",
    "mbpp/827": "Sum that index across the rows.",
    "mbpp/925": "A running product over the tuple.",
    "mbpp/10": "Sort and take the first n.",
    "mbpp/680": "Every element has to exceed the one before it.",
    "mbpp/697": "Count the elements with a zero remainder.",
    "mbpp/644": "Reverse the first k elements through a slice.",
    "mbpp/727": "Strip everything outside [a-zA-Z0-9] with one substitution.",
    "mbpp/586": "Rotate by slicing: the tail first, then the head.",
    "mbpp/647": "Each piece is an uppercase letter followed by everything up to the next one.",
    "mbpp/811": "List equality already compares element by element.",
    "mbpp/6": "XOR them and check that exactly one bit is set.",
    "mbpp/636": "Reciprocal roots means their product, c/a, is 1.",
})

# first-pass framings for tasks the base model fails (their verified fixed code becomes the first-pass target)
FRAMING.update({
    "mbpp/765": "Polite numbers are exactly the integers that aren't powers of two, so count those from 3 up.",
    "mbpp/608": "Build the Bell triangle row by row; each row starts with the previous row's last value.",
    "mbpp/918": "Classic coin-change count: one pass over the coins, ways[i] += ways[i - c].",
    "mbpp/912": "Lobb numbers have a closed form, (2m + 1)/(m + n + 1) · C(2n, m + n).",
    "mbpp/638": "The standard wind chill index, rounded to an int; velocity comes first in the call.",
    "mbpp/814": "The two arguments are the diagonals, so the area is p·q / 2.",
    "mbpp/910": "Coerce to ints and let `datetime.date` decide what's valid.",
    "mbpp/934": "Delannoy numbers are a single sum, Σₖ C(m, k)·C(n, k)·2ᵏ.",
    "mbpp/952": "Python ints are exact, so compute C(n, r) and reduce once; p isn't always prime here.",
    "mbpp/926": "Exactly k fixed points means choose them, then derange the rest: C(n, k) · !(n − k).",
    "mbpp/958": "Greedy over a value table that includes the subtractive pairs (CM, XC, IV, …).",
    "mbpp/830": "Scale up by 10**digits, ceil, scale back down.",
    "mbpp/738": "The series is 1 + 1/2 + … + 1/2ⁿ.",
    "mbpp/519": "One edge means a regular tetrahedron: V = a³ / (6√2).",
    "mbpp/799": "Rotate within a fixed 32-bit word.",
    "mbpp/927": "Height is 1 + the taller subtree; the tests also need a `Node` class.",
    "mbpp/671": "`n | (n + 1)` sets the lowest zero bit; all-ones numbers stay as they are.",
    "mbpp/853": "Divide out the factors of 2; the divisors of what's left are exactly the odd factors.",
    "mbpp/677": "The inputs are angles, so check they're positive and sum to 180.",
    "mbpp/851": "Σ 1/d over the divisors of n is σ(n)/n, and the second argument is σ(n).",
    "mbpp/768": "Odd parity means an odd count of set bits.",
    "mbpp/511": "The smallest factor sum comes from the prime factorisation, counted with multiplicity.",
    "mbpp/581": "Base area plus four triangular faces: b² + 2·b·s with slant height s.",
    "mbpp/957": "`n & -n` isolates the lowest set bit; its bit length is the position.",
    "mbpp/603": "Ludic sieve: take the first survivor k, then delete every k-th element starting from it.",
    "mbpp/556": "A pair's XOR is odd exactly when one is odd and one is even, so it's odd × even.",
    "mbpp/793": "Scan from the right; the arrays aren't reliably sorted.",
    "mbpp/907": "Lucky-number sieve over the odd numbers, deleting every k-th survivor.",
    "mbpp/668": "A regex collapses runs of the character into one.",
    "mbpp/531": "Greedy fails on coins like [9, 6, 5, 1], so a min-coins DP over values.",
    "mbpp/684": "The string repeats out to 10 characters: whole copies plus a partial prefix.",
    "mbpp/841": "Count pairs i < j with arr[i] > arr[j]; n is small enough for the direct loop.",
    "mbpp/536": "Every nth item is a slice with a step.",
    "mbpp/968": "floor(A·x/B) − A·floor(x/B) peaks at x = min(B − 1, N).",
    "mbpp/515": "Track which remainders mod m subset sums can reach and stop at 0.",
    "mbpp/640": "The input is a one-element list; strip the parenthesised part with a regex.",
    "mbpp/734": "Running recurrence: the products ending at i are arr[i] · (previous + 1).",
    "mbpp/663": "Step down from n to the nearest value with the right residue.",
    "mbpp/953": "Each subset holds a value once, so you need as many subsets as the top frequency.",
    "mbpp/626": "The largest triangle in a semicircle has area r²; negative radii get -1.",
    "mbpp/735": "XOR with a mask covering the bits strictly between the top bit and bit 0.",
    "mbpp/546": "`rfind` gives the last position; the expected output is 1-based, None when absent.",
    "mbpp/602": "One pass with a seen-set; the first character already seen is the answer.",
    "mbpp/787": "Search for `ab{3}` anywhere in the string.",
    "mbpp/767": "Count with a Counter and take each unordered pair once.",
    "mbpp/537": "One pass with a seen-set over the words.",
    "mbpp/650": "Equal here means the same elements in any order, so compare sorted copies.",
    "mbpp/856": "Each 1 must cross every 0 to its right; count those from the right.",
    "mbpp/719": "Search for `ab*` anywhere in the string.",
    "mbpp/965": "Insert an underscore before each inner capital, then lower-case.",
    "mbpp/791": "Keep everything that isn't itself a tuple.",
    "mbpp/954": "Profit is amount minus cost, when positive.",
    "mbpp/5": "Odd widths can't be tiled; even widths follow A(n) = 4·A(n−2) − A(n−4).",
    "mbpp/620": "Sorted, a pairwise-divisible subset is a divisor chain, so a longest-chain DP.",
    "mbpp/908": "First index whose value equals the index.",
    "mbpp/974": "Bottom-up: each cell adds the smaller of its two children.",
    "mbpp/656": "Pairing both arrays in sorted order minimises the total difference.",
    "mbpp/836": "Kadane's algorithm, tracking where the current run started.",
    "mbpp/575": "Walk the range and return the nth value that isn't a multiple.",
    "mbpp/867": "If the sum is odd, add 1; otherwise the smallest positive fix is 2.",
    "mbpp/838": "Count both mismatch types; same-type pairs take one swap, a mixed leftover takes two.",
    "mbpp/597": "Merge the two sorted arrays lazily and index the kth.",
    "mbpp/756": "Search for `ab?` anywhere in the string.",
    "mbpp/806": "Track the current uppercase run and the best seen.",
    "mbpp/788": "Append the string as one element and convert to a tuple.",
    "mbpp/528": "`min` with `key=len`, returned alongside its length.",
    "mbpp/547": "Consecutive numbers differ in the bits of `i ^ (i + 1)`.",
    "mbpp/572": "Keep the values that appear exactly once, in their original order.",
    "mbpp/2": "Set intersection, returned as a sorted tuple.",
})
