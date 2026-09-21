#!/usr/bin/env python3
"""Put Hohe ASR on AWS, on spot, behind a load balancer. Idempotent.

    ./launch.py --up            # create or update everything, print the address
    ./launch.py --status
    ./launch.py --restart       # replace the machine (after new code or weights)
    ./launch.py --down          # remove it all

The shape, and why:

**Spot, with an Auto Scaling Group.** Amazon takes the machine back with two
minutes of notice; the group notices and starts another. That group IS the
watcher, already written and already tested, so we write none. Interruption
costs a few minutes, which is acceptable here because the bot queues clips and
delivers them afterwards rather than failing.

**Internal load balancer, not public.** The only client is the bot on the app
server, in this same VPC. Keeping the balancer internal means no certificate to
issue, no DNS to own, nothing exposed to the internet, and audio that never
leaves Amazon's private network. It also gives the bot one stable address
across every spot replacement, which is the whole point.

**An instance profile, not a key.** The profile named in config.json has to
read the bucket holding the weights and the SSM parameter holding the shared
secret. Nothing else is granted, and no key travels in the user data.

The browser demo later needs a public address and a certificate. That is a
listener and a DNS record added to this, not a different design.
"""
from __future__ import annotations

import argparse
import base64
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Everything particular to one AWS account lives in config.json, which is not
#: committed. Copy config.example.json, fill in your own network, and nothing
#: else in this file changes. Without it this script says what is missing
#: rather than creating things in somebody else's VPC.
CONFIG_PATH = HERE / "config.json"
if not CONFIG_PATH.exists():
    sys.exit("no deploy/config.json: copy deploy/config.example.json and fill in your own values")
CFG = json.loads(CONFIG_PATH.read_text())

PROFILE, REGION = CFG["profile"], CFG["region"]
VPC, SUBNETS = CFG["vpc"], CFG["subnets"]
#: The only thing allowed to talk to the model: the security group of whatever
#: calls it, which for us is the server the bot runs on.
CLIENT_SG = CFG["client_security_group"]
PROFILE_NAME = CFG["instance_profile"]
BUCKET = CFG["bucket"]
MODEL_PREFIX = CFG.get("model_prefix", "asr/v0.3")
CODE_KEY = CFG.get("code_key", "hohe/serve.py")
SECRET_PARAM = CFG.get("secret_param", "/hohe/asr-secret")
NAME = CFG.get("name", "hohe-asr")
PORT = int(CFG.get("port", 8080))
#: Ordered by preference. Several families and sizes so the spot pool is deep
#: enough that we are rarely without capacity; capacity-optimized picks whichever
#: is least likely to be interrupted right now.
TYPES = ["c7i.xlarge", "c6i.xlarge", "c7a.xlarge", "m7i.xlarge", "c6a.xlarge"]
#: One machine is the resting state and what the bot needs. The ceiling exists
#: for the public demo: a link that does well puts a crowd behind a service that
#: takes one clip at a time, and the failure is not abuse, it is being slow in
#: front of the largest audience we will get. Five machines is about $11 for a
#: busy day and they go away again on their own.
MAX_MACHINES = 5
#: Each machine works one clip at a time, so busy really does mean busy. 55%
#: average adds a machine while people are still being answered quickly rather
#: than after they have given up.
TARGET_CPU = 55.0
THREADS = 4  # measured: eight is no faster and slower on short clips, see BENCH.md


def aws(*a: str, check: bool = True) -> str:
    p = subprocess.run(["aws", "--profile", PROFILE, "--region", REGION, *a], capture_output=True, text=True)
    if check and p.returncode:
        sys.exit(f"aws {' '.join(a[:3])}: {p.stderr.strip()}")
    return p.stdout.strip()


def j(*a: str, check: bool = True):
    out = aws(*a, "--output", "json", check=check)
    return json.loads(out) if out else None


def ami() -> str:
    return aws("ssm", "get-parameter", "--name",
               "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
               "--query", "Parameter.Value", "--output", "text")


def sg(name: str, desc: str) -> str:
    got = j("ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={name}",
            f"Name=vpc-id,Values={VPC}", check=False)
    if got and got["SecurityGroups"]:
        return got["SecurityGroups"][0]["GroupId"]
    return aws("ec2", "create-security-group", "--group-name", name, "--description", desc,
               "--vpc-id", VPC, "--query", "GroupId", "--output", "text")


def allow(group: str, source: str) -> None:
    aws("ec2", "authorize-security-group-ingress", "--group-id", group, "--protocol", "tcp",
        "--port", str(PORT), "--source-group", source, check=False)


def secret() -> str:
    got = aws("ssm", "get-parameter", "--name", SECRET_PARAM, "--with-decryption",
              "--query", "Parameter.Value", "--output", "text", check=False)
    if got:
        return got
    value = secrets.token_urlsafe(32)
    aws("ssm", "put-parameter", "--name", SECRET_PARAM, "--type", "SecureString", "--value", value)
    print(f"made a new shared secret at {SECRET_PARAM}; set ASR_SECRET on the bot from there")
    return value


def user_data() -> str:
    t = (HERE / "user_data.sh").read_text()
    for k, v in {"@@BUCKET@@": BUCKET, "@@MODEL_PREFIX@@": MODEL_PREFIX, "@@CODE_KEY@@": CODE_KEY,
                 "@@SECRET_PARAM@@": SECRET_PARAM, "@@THREADS@@": str(THREADS)}.items():
        t = t.replace(k, v)
    assert "@@" not in t, "unfilled placeholder in user_data.sh"
    return base64.b64encode(t.encode()).decode()


def up() -> None:
    # The code the machine runs travels through S3, like the weights, so a
    # replacement always starts from exactly what was last deployed.
    aws("s3", "cp", str(HERE.parent / "serve.py"), f"s3://{BUCKET}/{CODE_KEY}")
    secret()

    alb_sg = sg(f"{NAME}-alb", "Hohe ASR load balancer")
    inst_sg = sg(f"{NAME}-inst", "Hohe ASR machines")
    allow(alb_sg, CLIENT_SG)     # only the bot's server may reach the balancer
    allow(inst_sg, alb_sg)       # only the balancer may reach the machines

    lt = j("ec2", "describe-launch-templates", "--filters", f"Name=launch-template-name,Values={NAME}", check=False)
    data = {"ImageId": ami(), "IamInstanceProfile": {"Name": PROFILE_NAME},
            "SecurityGroupIds": [inst_sg], "UserData": user_data(),
            "BlockDeviceMappings": [{"DeviceName": "/dev/xvda",
                                     "Ebs": {"VolumeSize": 30, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            # Protected=true, because the reaper kills anything whose CPU has been
            # quiet for half an hour, and a transcription service is quiet by
            # design: it does nothing at all between voice notes. Without this
            # tag the reaper and the scaling group fight forever, the reaper
            # terminating an idle machine and the group booting another, and
            # every round costs five minutes of the service being down.
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": NAME}, {"Key": "Protected", "Value": "true"}]}],
            "MetadataOptions": {"HttpTokens": "required"}}
    if lt and lt["LaunchTemplates"]:
        aws("ec2", "create-launch-template-version", "--launch-template-name", NAME,
            "--source-version", "$Latest", "--launch-template-data", json.dumps(data))
    else:
        aws("ec2", "create-launch-template", "--launch-template-name", NAME,
            "--launch-template-data", json.dumps(data))

    tg = j("elbv2", "describe-target-groups", "--names", NAME, check=False)
    if tg and tg["TargetGroups"]:
        tg_arn = tg["TargetGroups"][0]["TargetGroupArn"]
    else:
        tg_arn = j("elbv2", "create-target-group", "--name", NAME, "--protocol", "HTTP", "--port", str(PORT),
                   "--vpc-id", VPC, "--target-type", "instance", "--health-check-path", "/health",
                   # Loading 2.4 GB takes a minute or two, so the grace is wide.
                   "--health-check-interval-seconds", "15", "--healthy-threshold-count", "2",
                   "--unhealthy-threshold-count", "4",
                   )["TargetGroups"][0]["TargetGroupArn"]

    lb = j("elbv2", "describe-load-balancers", "--names", NAME, check=False)
    if lb and lb["LoadBalancers"]:
        lb_arn, dns = lb["LoadBalancers"][0]["LoadBalancerArn"], lb["LoadBalancers"][0]["DNSName"]
    else:
        made = j("elbv2", "create-load-balancer", "--name", NAME, "--type", "application",
                 "--scheme", "internal", "--security-groups", alb_sg, "--subnets", *SUBNETS)["LoadBalancers"][0]
        lb_arn, dns = made["LoadBalancerArn"], made["DNSName"]
        aws("elbv2", "create-listener", "--load-balancer-arn", lb_arn, "--protocol", "HTTP",
            "--port", str(PORT), "--default-actions", f"Type=forward,TargetGroupArn={tg_arn}")

    asg = j("autoscaling", "describe-auto-scaling-groups", "--auto-scaling-group-names", NAME, check=False)
    policy = {"LaunchTemplate": {
        "LaunchTemplateSpecification": {"LaunchTemplateName": NAME, "Version": "$Latest"},
        "Overrides": [{"InstanceType": t} for t in TYPES]},
        # Entirely spot. capacity-optimized picks the deepest pool, which is the
        # single biggest lever on how often we are interrupted.
        "InstancesDistribution": {"OnDemandBaseCapacity": 0, "OnDemandPercentageAboveBaseCapacity": 0,
                                  "SpotAllocationStrategy": "capacity-optimized"}}
    if asg and asg["AutoScalingGroups"]:
        aws("autoscaling", "update-auto-scaling-group", "--auto-scaling-group-name", NAME,
            "--mixed-instances-policy", json.dumps(policy), "--min-size", "1",
            "--max-size", str(MAX_MACHINES))
    else:
        aws("autoscaling", "create-auto-scaling-group", "--auto-scaling-group-name", NAME,
            "--mixed-instances-policy", json.dumps(policy), "--min-size", "1",
            "--max-size", str(MAX_MACHINES),
            "--desired-capacity", "1", "--vpc-zone-identifier", ",".join(SUBNETS),
            "--target-group-arns", tg_arn, "--health-check-type", "ELB",
            # Long enough for a boot that installs torch and copies the weights.
            "--health-check-grace-period", "600",
            "--tags", f"Key=Name,Value={NAME},PropagateAtLaunch=true")
    # Target tracking rather than step scaling: one rule, no alarms to keep in
    # sync, and AWS handles the cooldowns. Scaling in is slow on purpose, so a
    # lull between two waves of visitors does not throw away a warm machine
    # that takes five minutes to replace.
    aws("autoscaling", "put-scaling-policy", "--auto-scaling-group-name", NAME,
        "--policy-name", f"{NAME}-cpu", "--policy-type", "TargetTrackingScaling",
        "--estimated-instance-warmup", "420",
        "--target-tracking-configuration", json.dumps({
            "PredefinedMetricSpecification": {"PredefinedMetricType": "ASGAverageCPUUtilization"},
            "TargetValue": TARGET_CPU, "DisableScaleIn": False}))

    print(f"\nASR_URL=http://{dns}:{PORT}")
    print(f"scales 1 to {MAX_MACHINES} machines at {TARGET_CPU:.0f}% average cpu")
    print(f"ASR_SECRET: read it from SSM at {SECRET_PARAM}")


def status() -> None:
    asg = j("autoscaling", "describe-auto-scaling-groups", "--auto-scaling-group-names", NAME, check=False)
    if not asg or not asg["AutoScalingGroups"]:
        return print("not deployed")
    g = asg["AutoScalingGroups"][0]
    for i in g["Instances"]:
        print(f"  {i['InstanceId']}  {i['InstanceType']}  {i['LifecycleState']}  health={i['HealthStatus']}")
    lb = j("elbv2", "describe-load-balancers", "--names", NAME, check=False)
    if lb and lb["LoadBalancers"]:
        print(f"  address http://{lb['LoadBalancers'][0]['DNSName']}:{PORT}")
    tg = j("elbv2", "describe-target-groups", "--names", NAME, check=False)
    if tg and tg["TargetGroups"]:
        h = j("elbv2", "describe-target-health", "--target-group-arn", tg["TargetGroups"][0]["TargetGroupArn"])
        for t in h["TargetHealthDescriptions"]:
            print(f"  target {t['Target']['Id']}: {t['TargetHealth']['State']} "
                  f"{t['TargetHealth'].get('Description', '')}")


def restart() -> None:
    """New code or new weights: replace the machine, which re-runs user_data."""
    aws("autoscaling", "start-instance-refresh", "--auto-scaling-group-name", NAME,
        "--preferences", json.dumps({"MinHealthyPercentage": 0, "InstanceWarmup": 600}))
    print("replacing the machine; watch with --status")


def down() -> None:
    aws("autoscaling", "delete-auto-scaling-group", "--auto-scaling-group-name", NAME, "--force-delete", check=False)
    lb = j("elbv2", "describe-load-balancers", "--names", NAME, check=False)
    if lb and lb["LoadBalancers"]:
        aws("elbv2", "delete-load-balancer", "--load-balancer-arn", lb["LoadBalancers"][0]["LoadBalancerArn"])
        time.sleep(20)
    tg = j("elbv2", "describe-target-groups", "--names", NAME, check=False)
    if tg and tg["TargetGroups"]:
        aws("elbv2", "delete-target-group", "--target-group-arn", tg["TargetGroups"][0]["TargetGroupArn"], check=False)
    aws("ec2", "delete-launch-template", "--launch-template-name", NAME, check=False)
    print("removed (security groups and the secret are left alone)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("up", "status", "restart", "down"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.up:
        up()
    elif a.restart:
        restart()
    elif a.down:
        down()
    elif a.status:
        status()
    else:
        ap.print_help()
