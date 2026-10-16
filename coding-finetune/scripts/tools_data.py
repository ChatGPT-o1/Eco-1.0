"""Tool-use scenarios (train + held-out eval) in the Qwen3.5 tool format.

Each scenario: tools (subset of TOOLS), messages (context), and either
  target: {"thought", "name", "arguments"}  -> assistant tool call
  answer: str                               -> plain final answer (no call)
Eval cases use `expect` with matchers instead of a target:
  value       exact match (strings compared after strip / whitespace squash)
  {"$any": 1} parameter must be present and non-empty
  {"$contains": s} substring match
"""


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}
A = {"type": "array", "items": {"type": "string"}}

TOOLS = {t["function"]["name"]: t for t in [
    fn("read_file", "Read a text file from the workspace.", {"path": S}, ["path"]),
    fn("write_file", "Create or overwrite a file.", {"path": S, "content": S}, ["path", "content"]),
    fn("edit_file", "Replace one exact occurrence of old_str with new_str in a file.",
       {"path": S, "old_str": S, "new_str": S}, ["path", "old_str", "new_str"]),
    fn("run_command", "Run a shell command in the workspace and return stdout/stderr.",
       {"command": S, "timeout": {"type": "integer", "description": "seconds"}}, ["command"]),
    fn("search_code", "Regex search across the repository.", {"pattern": S, "path": S}, ["pattern"]),
    fn("list_dir", "List entries in a directory.", {"path": S}, ["path"]),
    fn("run_tests", "Run the test suite or a subset.", {"path": S, "keyword": S, "verbose": B}, []),
    fn("git_commit", "Stage all changes and commit.", {"message": S}, ["message"]),
    fn("http_get", "Fetch a URL and return the body.", {"url": S}, ["url"]),
    fn("create_issue", "Open an issue in the tracker.", {"title": S, "body": S, "labels": A}, ["title"]),
    fn("get_weather", "Current weather for a city.", {"city": S, "units": {"type": "string", "enum": ["metric", "imperial"]}}, ["city"]),
    fn("send_message", "Send a chat message to a channel.", {"channel": S, "text": S}, ["channel", "text"]),
    fn("sql_query", "Run a read-only SQL query against the app database.", {"query": S, "limit": I}, ["query"]),
    fn("install_package", "Install a Python package with pip.", {"name": S, "version": S}, ["name"]),
    fn("set_timer", "Start a countdown timer.", {"minutes": I, "label": S}, ["minutes"]),
    # eval-only tools (never seen in training)
    fn("grep_repo", "Search files for a literal string.", {"query": S, "glob": S}, ["query"]),
    fn("apply_patch", "Apply a unified diff to the workspace.", {"patch": S}, ["patch"]),
    fn("docker_logs", "Show recent logs for a container.", {"container": S, "tail": I}, ["container"]),
    fn("translate", "Translate text to a target language.", {"text": S, "target_lang": S}, ["text", "target_lang"]),
    fn("calendar_add", "Add a calendar event.", {"title": S, "date": S, "time": S, "attendees": A}, ["title", "date"]),
]}


def T(*names):
    return [TOOLS[n] for n in names]


def call(_fn, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": _fn, "arguments": args}}]}


def thought_call(thought, _fn, **args):
    m = call(_fn, **args)
    m["content"] = thought
    return m


def tool(text):
    return {"role": "tool", "content": text}


def U(text):
    return {"role": "user", "content": text}


DEV = T("read_file", "edit_file", "run_command", "search_code", "list_dir", "run_tests", "git_commit")

TRAIN = [
    # --- single calls, arguments straight from the request
    dict(tools=DEV, messages=[U("what's in src/config.py?")],
         target=("Opening it.", "read_file", {"path": "src/config.py"})),
    dict(tools=DEV, messages=[U("run the tests for the parser module")],
         target=("Running just the parser tests.", "run_tests", {"path": "tests/test_parser.py"})),
    dict(tools=DEV, messages=[U("where do we call fetch_user?")],
         target=("Searching for call sites.", "search_code", {"pattern": r"fetch_user\("})),
    dict(tools=DEV, messages=[U("show me what's under app/routes")],
         target=("Listing it.", "list_dir", {"path": "app/routes"})),
    dict(tools=DEV, messages=[U("check the python version on this box")],
         target=("Quick check.", "run_command", {"command": "python --version"})),
    dict(tools=DEV, messages=[U("run pytest with a 2 minute timeout, the suite sometimes hangs")],
         target=("Capping it at 120 seconds so a hang can't stall us.", "run_command", {"command": "pytest -q", "timeout": 120})),
    dict(tools=DEV, messages=[U("run only the tests matching 'auth', verbose please")],
         target=("Filtering to the auth tests with verbose output.", "run_tests", {"keyword": "auth", "verbose": True})),
    dict(tools=DEV, messages=[U("commit this as 'fix off-by-one in paginator'")],
         target=("Committing.", "git_commit", {"message": "fix off-by-one in paginator"})),
    dict(tools=T("http_get", "read_file"), messages=[U("grab https://api.example.com/health and tell me if it's up")],
         target=("Hitting the health endpoint.", "http_get", {"url": "https://api.example.com/health"})),
    dict(tools=T("create_issue", "search_code"), messages=[U("file a bug: login button does nothing on Safari. label it bug and frontend")],
         target=("Filing it with both labels.", "create_issue",
                 {"title": "Login button does nothing on Safari", "body": "Clicking the login button on Safari has no effect.",
                  "labels": ["bug", "frontend"]})),
    dict(tools=T("get_weather", "send_message"), messages=[U("weather in Lisbon? I think in fahrenheit")],
         target=("Pulling it in imperial units.", "get_weather", {"city": "Lisbon", "units": "imperial"})),
    dict(tools=T("sql_query"), messages=[U("how many users signed up today? just count rows in users where created_at is today")],
         target=("Counting today's signups.", "sql_query",
                 {"query": "SELECT COUNT(*) FROM users WHERE DATE(created_at) = CURRENT_DATE"})),
    dict(tools=T("sql_query"), messages=[U("show me the 5 most recent orders")],
         target=("Newest five first.", "sql_query", {"query": "SELECT * FROM orders ORDER BY created_at DESC", "limit": 5})),
    dict(tools=T("install_package", "run_command"), messages=[U("install requests 2.31.0")],
         target=("Pinning that exact version.", "install_package", {"name": "requests", "version": "2.31.0"})),
    dict(tools=T("set_timer", "send_message"), messages=[U("25 min pomodoro timer pls")],
         target=("Starting it.", "set_timer", {"minutes": 25, "label": "pomodoro"})),
    dict(tools=T("send_message", "get_weather"), messages=[U("tell #deploys that the release is out")],
         target=("Posting it.", "send_message", {"channel": "#deploys", "text": "The release is out."})),
    dict(tools=DEV, messages=[U("rename the variable cnt to count in utils/stats.py, it's only on the line `cnt = 0`")],
         target=("That's a single exact line, so one edit does it.", "edit_file",
                 {"path": "utils/stats.py", "old_str": "cnt = 0", "new_str": "count = 0"})),
    dict(tools=DEV, messages=[U("the build is failing, can you see why?")],
         target=("Reproducing it first so we work from the real error.", "run_command", {"command": "make build"})),
    dict(tools=DEV, messages=[U("is there a TODO left anywhere in src?")],
         target=("Searching src for leftovers.", "search_code", {"pattern": "TODO", "path": "src"})),

    # --- error -> pivot (failed earlier call is context only)
    dict(tools=DEV, messages=[U("open the settings file"), call("read_file", path="settings.py"),
                              tool("Error: settings.py not found")],
         target=("Not at the root; listing to find where it lives.", "list_dir", {"path": "."})),
    dict(tools=DEV, messages=[U("open the settings file"), call("read_file", path="settings.py"),
                              tool("Error: settings.py not found"), call("list_dir", path="."),
                              tool("README.md\napp/\ntests/\nrequirements.txt")],
         target=("It's likely inside the app package.", "list_dir", {"path": "app"})),
    dict(tools=DEV, messages=[U("open the settings file"), call("read_file", path="settings.py"),
                              tool("Error: settings.py not found"), call("list_dir", path="app"),
                              tool("__init__.py\nsettings.py\nviews.py")],
         target=("Found it under app/.", "read_file", {"path": "app/settings.py"})),
    dict(tools=DEV, messages=[U("run the tests"), call("run_command", command="pytest"),
                              tool("bash: pytest: command not found")],
         target=("pytest isn't on PATH; running it through the interpreter instead.", "run_command",
                 {"command": "python -m pytest -q"})),
    dict(tools=DEV, messages=[U("fix the failing test"), call("run_tests"),
                              tool("FAILED tests/test_math.py::test_mean - ZeroDivisionError: division by zero\n"
                                   "  src/mathx.py:4: return sum(xs) / len(xs)")],
         target=("`mean` divides by `len(xs)` with no empty-list guard; reading it before editing.", "read_file",
                 {"path": "src/mathx.py"})),
    dict(tools=DEV, messages=[U("fix the failing test"), call("run_tests"),
                              tool("FAILED tests/test_math.py::test_mean - ZeroDivisionError: division by zero\n"
                                   "  src/mathx.py:4: return sum(xs) / len(xs)"),
                              call("read_file", path="src/mathx.py"),
                              tool("def mean(xs):\n    return sum(xs) / len(xs)\n"),
                              U("the test expects mean([]) == 0")],
         target=("Guarding the empty case where the division happens.", "edit_file",
                 {"path": "src/mathx.py", "old_str": "    return sum(xs) / len(xs)",
                  "new_str": "    return sum(xs) / len(xs) if xs else 0"})),
    dict(tools=DEV, messages=[U("fix the failing test"), call("edit_file", path="src/mathx.py",
                              old_str="    return sum(xs) / len(xs)", new_str="    return sum(xs) / len(xs) if xs else 0"),
                              tool("ok")],
         target=("Re-running the suite to confirm.", "run_tests", {"path": "tests/test_math.py"})),
    dict(tools=DEV, messages=[U("bump the version to 1.4.0 in pyproject.toml"),
                              call("edit_file", path="pyproject.toml", old_str='version = "1.3"', new_str='version = "1.4.0"'),
                              tool("Error: old_str not found in pyproject.toml")],
         target=("My guess at the old line was wrong; reading the file to get the exact text.", "read_file",
                 {"path": "pyproject.toml"})),
    dict(tools=DEV, messages=[U("bump the version to 1.4.0 in pyproject.toml"),
                              call("edit_file", path="pyproject.toml", old_str='version = "1.3"', new_str='version = "1.4.0"'),
                              tool("Error: old_str not found in pyproject.toml"),
                              call("read_file", path="pyproject.toml"),
                              tool('[project]\nname = "eco"\nversion = "1.3.2"\n')],
         target=("The current version is 1.3.2.", "edit_file",
                 {"path": "pyproject.toml", "old_str": 'version = "1.3.2"', "new_str": 'version = "1.4.0"'})),
    dict(tools=DEV, messages=[U("find where we define parse_date"), call("search_code", pattern="def parse_date"),
                              tool("No matches.")],
         target=("No plain def; it may be a lambda or an attribute, so widening the search.", "search_code",
                 {"pattern": "parse_date\\s*="})),
    dict(tools=T("http_get"), messages=[U("check https://status.example.com"), call("http_get", url="https://status.example.com"),
                                        tool("Error: timed out after 10s")],
         target=("Timed out once; retrying the lighter health path before calling it down.", "http_get",
                 {"url": "https://status.example.com/health"})),
    dict(tools=T("install_package", "run_command"), messages=[U("install numpy 1.19 for this project"),
                                                              call("install_package", name="numpy", version="1.19"),
                                                              tool("ERROR: No matching distribution found for numpy==1.19 (Python 3.12)")],
         target=("numpy 1.19 has no wheels for Python 3.12; checking which interpreter the project expects.", "run_command",
                 {"command": "cat .python-version"})),
    dict(tools=T("sql_query"), messages=[U("how many orders last week?"),
                                         call("sql_query", query="SELECT COUNT(*) FROM order WHERE created_at > NOW() - INTERVAL '7 days'"),
                                         tool("ERROR: syntax error at or near \"order\"")],
         target=("`order` is a reserved word; the table is `orders`.", "sql_query",
                 {"query": "SELECT COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '7 days'"})),

    # --- knowing when to stop: answer from tool output, no extra call
    dict(tools=T("http_get"), messages=[U("grab https://api.example.com/health and tell me if it's up"),
                                        call("http_get", url="https://api.example.com/health"),
                                        tool('{"status": "ok", "uptime_s": 86400}')],
         answer="It's up: status ok, running for a full day."),
    dict(tools=DEV, messages=[U("run the tests"), call("run_command", command="python -m pytest -q"),
                              tool("........................ 24 passed in 1.92s")],
         answer="All 24 tests pass."),
    dict(tools=T("get_weather"), messages=[U("weather in Oslo?"), call("get_weather", city="Oslo"),
                                           tool('{"temp_c": -3, "conditions": "light snow"}')],
         answer="-3°C with light snow in Oslo."),
    dict(tools=DEV, messages=[U("fix the failing test"), call("run_tests", path="tests/test_math.py"),
                              tool("3 passed in 0.12s")],
         answer="Fixed: `mean` now returns 0 for an empty list, and tests/test_math.py passes."),
    dict(tools=T("sql_query"), messages=[U("how many users signed up today?"),
                                         call("sql_query", query="SELECT COUNT(*) FROM users WHERE DATE(created_at) = CURRENT_DATE"),
                                         tool("count\n-----\n  137")],
         answer="137 signups today."),
    dict(tools=T("get_weather", "send_message"), messages=[U("what's 17 * 23?")],
         answer="391."),
    dict(tools=DEV, messages=[U("what does `git stash` do?")],
         answer="It shelves your uncommitted changes and resets the working tree to HEAD; `git stash pop` brings them back."),
]

EVAL = [
    dict(id="t1", tools=DEV, messages=[U("open tests/conftest.py")],
         expect={"name": "read_file", "arguments": {"path": "tests/conftest.py"}}),
    dict(id="t2", tools=DEV, messages=[U("run the test suite but give up after 300 seconds")],
         expect={"name": "run_command", "arguments": {"command": {"$contains": "pytest"}, "timeout": 300}}),
    dict(id="t3", tools=DEV, messages=[U("run just the tests with 'billing' in the name, verbose")],
         expect={"name": "run_tests", "arguments": {"keyword": "billing", "verbose": True}}),
    dict(id="t4", tools=T("create_issue", "send_message"),
         messages=[U("open an issue titled 'Crash on empty cart' with labels bug and checkout")],
         expect={"name": "create_issue", "arguments": {"title": "Crash on empty cart", "labels": ["bug", "checkout"]}}),
    dict(id="t5", tools=T("grep_repo", "read_file"), messages=[U("find every place that mentions STRIPE_KEY in .env files")],
         expect={"name": "grep_repo", "arguments": {"query": "STRIPE_KEY", "glob": {"$contains": ".env"}}}),
    dict(id="t6", tools=T("docker_logs", "run_command"), messages=[U("show the last 50 lines of logs for the api container")],
         expect={"name": "docker_logs", "arguments": {"container": "api", "tail": 50}}),
    dict(id="t7", tools=T("translate", "send_message"), messages=[U("translate 'the build is green' into German")],
         expect={"name": "translate", "arguments": {"text": "the build is green", "target_lang": {"$in": ["German", "german", "de", "DE", "Deutsch", "deutsch"]}}}),
    dict(id="t8", tools=T("calendar_add", "send_message"),
         messages=[U("add 'design review' on 2026-11-03 at 14:00 with ana and raj")],
         expect={"name": "calendar_add", "arguments": {"title": "design review", "date": "2026-11-03", "time": "14:00",
                                                       "attendees": ["ana", "raj"]}}),
    dict(id="t9", tools=DEV, messages=[U("open the logging config"), call("read_file", path="logging.yaml"),
                                       tool("Error: logging.yaml not found")],
         expect={"name": {"$in": ["list_dir", "search_code"]}, "arguments": {}}),
    dict(id="t10", tools=DEV, messages=[U("run the tests"), call("run_command", command="pytest"),
                                        tool("bash: pytest: command not found")],
         expect={"name": "run_command", "arguments": {"command": {"$contains": "python -m pytest"}}}),
    dict(id="t11", tools=T("sql_query"), messages=[U("count the rows in the user table"),
                                                   call("sql_query", query="SELECT COUNT(*) FROM user"),
                                                   tool("ERROR: syntax error at or near \"user\"")],
         expect={"name": "sql_query", "arguments": {"query": {"$contains": "users"}}}),
    dict(id="t12", tools=DEV, messages=[U("change MAX_RETRIES = 3 to 5 in app/net.py"),
                                        call("edit_file", path="app/net.py", old_str="MAX_RETRIES=3", new_str="MAX_RETRIES=5"),
                                        tool("Error: old_str not found in app/net.py")],
         expect={"name": {"$in": ["read_file", "edit_file"]}, "arguments": {"path": "app/net.py"},
                 "if_name": {"edit_file": {"old_str": "MAX_RETRIES = 3", "new_str": "MAX_RETRIES = 5"}}}),
    dict(id="t13", tools=T("apply_patch", "read_file"),
         messages=[U("apply this patch:\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2")],
         expect={"name": "apply_patch", "arguments": {"patch": {"$contains": "+a = 2"}}}),
    dict(id="t14", tools=T("set_timer", "send_message"), messages=[U("remind me in 10 minutes to check the oven")],
         expect={"name": "set_timer", "arguments": {"minutes": 10}}),
    dict(id="t15", tools=T("http_get"), messages=[U("is https://example.org up?"), call("http_get", url="https://example.org"),
                                                  tool("<html><title>Example Domain</title></html>")],
         expect={"name": None}),
    dict(id="t16", tools=T("get_weather", "send_message"), messages=[U("what's the capital of Canada?")],
         expect={"name": None}),
]
