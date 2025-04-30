import json
import httpx
from openai import OpenAI
from tqdm import tqdm

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.xty.app/v1",
    api_key="sk-xxx",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

def query_model(prompt, max_length=100, retries=3):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise
            
            
def build_prompt(q, q_type, q_domain):
    if q_type == "单选题":
        return f'你是一个中国的{q_domain}专家。\n这是一道{q_type}，请给出你的答案（只需给出单个选项），并进行解析，给出你的理由。\n' + \
                '题目：酸碱灭火器是一种内部装有65％的工业硫酸和碳酸氢钠的水溶液作灭火剂的灭火器，使用时，两种药液混合发生化学反应，产生二氧化碳压力气体，灭火剂在二氧化碳气体压力下喷出进行灭火。下列火灾中，适用酸碱灭火器扑救的是（）。\nA．天然气火灾\nB．金属钠火灾\nC．配电柜火灾\nD．纺织物火灾\n' + \
                '答案：D\n' + \
                '解析：酸碱灭火器是一种内部装有65％的工业硫酸和碳酸氢铀的水溶液作灭火剂的灭火器。使用时，两种药液混合发生化学反应，产生二氧化碳压力气体，灭火剂在二氧化碳气体压力下喷出进行灭火。该类灭火器适用于扑救A类物质的初起火灾，如木、竹、织物、纸张等燃烧的火灾。它不能用于扑救B类物质燃烧的火灾也不能用于扑救C类可燃气体D类轻金属火灾，同时也不能用于带电场合火灾的扑救。\n\n' + \
                f'题目：{q}\n' + \
                '答案：'
    elif q_type == "多选题":
        return f'你是一个中国的{q_domain}专家。\n这是一道{q_type}，请给出你的答案（只需给出选项），并进行解析，给出你的理由。\n' + \
                '题目：关于运行速度的说法，正确的有下列哪些选项？()\nA．运行速度是路面平整、潮湿、自由流状态下，行驶速度累计分布曲线上对应于85%分位值的速度\nB．公路设计应采用运行速度进行检验\nC．相邻路段运行速度之差应小于20km/h\nD．同一路段运行速度与设计速度之差应小于20km/h\n' + \
                '答案：ABC\n' + \
                '解析：A项，运行速度是路面平整、潮湿、自由流状态下，行驶速度累计分布曲线上对应于85%分位值的速度。运行速度考虑了公路上绝大多数驾驶员的交通心理需求，是随着公路路线不断变化的。BCD三项，根据《公路工程技术标准》(JTGB01—2014)第3.5.2条规定，公路设计应采用运行速度进行检验，相邻路段运行速度之差应小于20km/h，同一路段运行速度与设计速度之差宜小于20km/h。《公路项目安全性评价规范》(JTGB05—2015)中有关于运行速度的计算模型。\n\n' + \
                f'题目：{q}\n' + \
                '答案：'
    elif q_type == "判断题":
        return f'你是一个中国的{q_domain}专家。\n这是一道{q_type}，请给出你的答案（只需给出“正确”或“错误”），并进行解析，给出你的理由。\n' + \
                '题目：基本建设投资和更新改造投资的综合范围为总投资50万元以上（含50万元）的项目。\n' + \
                '答案：正确\n' + \
                '解析：基本建设投资指企业、事业、行政单位以扩大生产能力或工程效益为主要目的的新建、扩建工程及有关工作。其综合范围为总投资50万元以上（含50万元）的基本建设项目。更新改造投资指企业、事业单位对原有设施进行固定资产更新和技术改造，以及相应配套的工程和有关工作（不包括大修理和维护工程）。其综合范围为总投资50万元以上（含50万元）的更新改造项目。\n\n' + \
                f'题目：{q}\n' + \
                '答案：'
    raise Exception(f"{q_type} error")

file_path = "../QualBench_dataset.json"

with open(file_path, "r", encoding="utf8") as f:
    datas = json.load(f)

with open(f"res_gpt4o.jsonl", "a", encoding="utf8") as f:
    for item in tqdm(datas):
        prompt = build_prompt(item["question"], item["question_type"], item["domain"])
        resp = query_model(prompt)
        item["resp"] = resp
        f.write(json.dumps(item, ensure_ascii=False))
        f.write("\n")
        f.flush()