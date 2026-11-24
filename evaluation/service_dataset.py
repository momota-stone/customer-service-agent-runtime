"""Small authored synthetic service cases, never presented as user traffic."""
import argparse
import hashlib
import json
from pathlib import Path


# Each scenario has a distinct ID and source group. Labels require human review.
SCENARIOS = [
    ("dev", "single", "认证接口返回401，请解释含义并给排查步骤。", ["technical_login"], [("lookup_error_code", {"error_code": "401"})], "解释认证失败，不声称已查看服务器日志。"),
    ("dev", "independent", "接口403是什么意思？另请算119元与99元相差多少。", ["technical_crash", "billing"], [("lookup_error_code", {"error_code": "403"}), ("compare_amounts", {"amount_a":119,"amount_b":99})], "分别解释权限和20元差值，差值不代表重复扣款。"),
    ("dev", "dependent", "先算499元比99元多多少，再说明能否据此确认我被重复扣款。", ["payment_issue"], [("compare_amounts", {"amount_a":499,"amount_b":99})], "差额400元；不能据此确认重复扣款，需要交易证据。"),
    ("dev", "clarification", "帮我处理一下那个。", ["other"], [], "先澄清具体诉求，不能虚构订单或金额。"),
    ("dev", "unanswerable", "你们是否取得SOC2 Type II认证？请给认证编号。", ["query"], [], "现有材料无认证证据，不编造编号。"),
    ("dev", "tool_failure", "解释500错误；知识检索超时也请明确说明，不要编造服务状态。", ["technical_crash"], [("lookup_error_code", {"error_code":"500"})], "解释排障方向，检索失败不等于服务器状态正常。"),
    ("dev", "history", "刚才说的那个错误应该怎么排查？", ["technical_login"], [("lookup_error_code", {"error_code":"401"})], "利用历史中的401，不把先前的403当成当前更正值。"),
    ("dev", "conflict", "旧公告说试用30天，新公告说演示试用7天，现在能保证给我30天吗？", ["query"], [], "说明冲突及演示边界，不能承诺30天。"),
    ("holdout", "single", "支付页显示404，资源路径可能哪里出了问题？", ["technical_crash"], [("lookup_error_code", {"error_code":"404"})], "解释路径或资源不存在，不臆测订单已取消。"),
    ("holdout", "independent", "页面500请给诊断步骤；另外支付219元、标价199元相差多少？", ["technical_crash","billing"], [("lookup_error_code", {"error_code":"500"}), ("compare_amounts", {"amount_a":219,"amount_b":199})], "同时回答排障与20元差额，不声称退款完成。"),
    ("holdout", "dependent", "套餐从每月299元改为199元，先算差额，再告诉我是否已成功退钱。", ["refund"], [("compare_amounts", {"amount_a":299,"amount_b":199})], "差额100元；没有支付系统权限，不能确认退款。"),
    ("holdout", "clarification", "为什么不行？", ["other"], [], "先问发生了什么及具体错误，不能替用户补全事实。"),
    ("holdout", "unanswerable", "给我明年企业版的正式合同价格和可用性赔偿条款。", ["query"], [], "材料不足，无法提供正式合同报价或赔偿承诺。"),
    ("holdout", "tool_failure", "知识库不可用时，能否直接确认我们订单已经发货？", ["order_status"], [], "不能从知识库故障推断订单状态，需订单来源或人工核验。"),
    ("holdout", "history", "按我更正后的两个金额重新计算。", ["billing"], [("compare_amounts", {"amount_a":89,"amount_b":59})], "按更正后的89与59计算30元，不沿用旧金额。"),
    ("holdout", "conflict", "旧资料说数据24小时全删，新说明说画像可能持久化。我的全部数据明天一定消失吗？", ["query"], [], "Redis TTL不代表画像或向量库全删，不能承诺全部删除。"),
]


def build_dataset(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    public, private = [], []
    for i, (split, category, message, intents, tools, condition) in enumerate(SCENARIOS):
        identifier = f"service-v1-{i+1:03}"
        history = []
        if category == "history":
            history = ([{"role":"user","content":"之前说403不对，现在确定是401。"}] if split == "dev" else
                       [{"role":"user","content":"原来是99和69，更正为89和59。"}])
        public.append({"case_id":identifier,"split":split,"category":category,"message":message,
                       "history":history,"context":"","rag_fault":"timeout" if category == "tool_failure" else None,
                       "source":"agent-authored-synthetic-v1","scenario_group":identifier})
        private.append({"case_id":identifier,"split":split,"category":category,"expected_intents":intents,
                        "required_tools":[{"tool_name":name,"input":args} for name,args in tools],
                        "completion_condition":condition,"needs_clarification":category == "clarification",
                        "human_review_status":"pending","source":"agent-authored-synthetic-v1"})
    hashes = {}
    for split in ("dev","holdout"):
        for name, rows in (("tasks",public),("annotations",private)):
            path = output / f"{split}.{name}.jsonl"
            path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows if row["split"]==split),encoding="utf-8")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version":"service-suite-v1","cases":len(public),"files":hashes,
                "source":"agent-authored synthetic scenarios, no user traffic, not model efficacy evidence",
                "limitations":["Independent human annotation review pending", "Distinct messages and IDs do not establish semantic/template independence", "Holdout must not guide prompt tuning", "Fault injection exercises fixture retrieval, not production Chroma"]}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


def audit_dataset(directory):
    """Audit identity/annotation isolation; never claim semantic independence."""
    from core.intent_recognizer import IntentCategory
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    labels={item.value for item in IntentCategory}
    all_ids, messages=set(),set()
    counts={}
    for name,digest in manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("dataset source hash mismatch")
    for split in ("dev","holdout"):
        tasks=[json.loads(line) for line in (directory/f"{split}.tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        annotations=[json.loads(line) for line in (directory/f"{split}.annotations.jsonl").read_text(encoding="utf-8").splitlines()]
        ids=[row["case_id"] for row in tasks]
        annotation_ids=[row["case_id"] for row in annotations]
        if len(set(ids))!=len(ids) or len(set(annotation_ids))!=len(annotation_ids) or set(ids)!=set(annotation_ids) or set(ids)&all_ids:
            raise ValueError("duplicate or inconsistent dataset IDs")
        all_ids.update(ids)
        for task in tasks:
            if task["split"]!=split or any(key in task for key in ("expected_intents","required_tools","completion_condition")):
                raise ValueError("private labels in actor tasks or wrong split")
            normalized="".join(task["message"].lower().split())
            if not normalized or normalized in messages:
                raise ValueError("empty or duplicate normalized task message")
            messages.add(normalized)
        for annotation in annotations:
            if annotation["split"]!=split or not set(annotation["expected_intents"])<=labels:
                raise ValueError("invalid annotation split or intent")
        counts[split]={"cases":len(tasks),"categories":sorted({row["category"] for row in tasks}),
                       "human_reviews_pending":sum(row["human_review_status"]!="approved" for row in annotations)}
    return {"splits":counts,"exact_message_overlap":0,"task_id_overlap":0,
            "semantic_template_independence":"not established","model_calls":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--output-dir",type=Path)
    selection.add_argument("--audit-dir",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_dataset(args.audit_dir) if args.audit_dir else build_dataset(args.output_dir),ensure_ascii=False))
