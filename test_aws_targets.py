"""test_aws_targets.py -- target isolation for the AWS compute/exec routes.

Run: python3 test_aws_targets.py   (needs fastapi + pydantic; stubs aws_routes)

WHAT THIS PROTECTS
  Before targets existed, AWS_COMPUTE_NAME_TAG was one global shared by
  provision AND every exec verb, and aws_exec_routes hardcoded
  SERVICE="kalshiml". Launching a second box therefore retargeted stop /
  restart / update away from the live trading engine. The two properties
  below are the ones that must never regress:

    1. Omitting `target` behaves EXACTLY as before -- same name tag from the
       same unsuffixed env var, same kalshiml unit in every verb.
    2. A non-default target inherits NOTHING implicitly -- not the name tag,
       and above all not the IAM instance profile, which is the blast radius.
"""
import os, sys
from fastapi import HTTPException
os.environ["AWS_COMPUTE_NAME_TAG"] = "kalshiml-engine-2"   # mirrors live bridge
import aws_compute_routes as C
import aws_exec_routes as E

fails = []
def check(label, got, want):
    ok = got == want
    print(("PASS " if ok else "FAIL ") + label, "->", repr(got))
    if not ok:
        fails.append((label, got, want))

# 1. Default target must behave EXACTLY as before the change.
check("default name_tag reads unsuffixed env", C._name_tag(), "kalshiml-engine-2")
check("explicit kalshiml identical", C._name_tag("kalshiml"), "kalshiml-engine-2")
check("default verb uses kalshiml unit",
      E._verbs(None)["restart"],
      "systemctl restart kalshiml && sleep 3 && systemctl is-active kalshiml")
check("default bootstrap_log path",
      E._verbs(None)["bootstrap_log"], "tail -n {n} /var/log/kalshiml-bootstrap.log")
check("default env file", E._verbs(None)["env"],
      "cut -d= -f1 /etc/kalshiml.env 2>/dev/null | sort")

# 2. Nowcaster must NOT inherit the unsuffixed env var.
check("nowcaster falls back to its own default tag",
      C._name_tag("nowcaster"), "nowcaster-engine")
check("nowcaster verb uses its own unit",
      E._verbs("nowcaster")["restart"],
      "systemctl restart nowcaster && sleep 3 && systemctl is-active nowcaster")
check("nowcaster disk uses its own data dir",
      "/var/lib/nowcaster/*" in E._verbs("nowcaster")["disk"], True)
check("nowcaster git uses its own repo dir",
      E._verbs("nowcaster")["git"].startswith("cd /opt/nowcaster "), True)

# 3. Suffixed env override works and does not leak to the default.
os.environ["AWS_COMPUTE_NAME_TAG_NOWCASTER"] = "nowcaster-engine-1"
check("suffixed override applies", C._name_tag("nowcaster"), "nowcaster-engine-1")
check("default untouched by suffixed var", C._name_tag(), "kalshiml-engine-2")

# 4. Instance profile must not be shared across targets.
os.environ["AWS_COMPUTE_IAM_PROFILE"] = "Kml"
check("kalshiml profile", C._env_for("kalshiml", "AWS_COMPUTE_IAM_PROFILE"), "Kml")
check("nowcaster does NOT inherit Kml profile",
      C._env_for("nowcaster", "AWS_COMPUTE_IAM_PROFILE"), "")

# 5. Unknown target is rejected, not silently defaulted.
try:
    C._name_tag("../../etc")
    check("unknown target rejected", "no raise", "HTTPException")
except HTTPException as e:
    check("unknown target rejected", e.status_code, 400)

# 6. Verb name set is stable across targets.
check("verb names identical across targets",
      sorted(E._verbs(None)) == sorted(E._verbs("nowcaster")) == E.VERB_NAMES, True)
check("verb count", len(E.VERB_NAMES), 14)

print()
print("FAILURES:", len(fails))
sys.exit(1 if fails else 0)
