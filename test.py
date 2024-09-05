import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLamaExtractor:
    def __init__(self, model_name='/root/.cache/huggingface/hub/models--FlagAlpha--Llama2-Chinese-7b-Chat/snapshots/9c1693247d2d1f99807b83b5dc817d700a3f2fa5'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cpu"
        self.model.to(device)

    def extract_information(self, text, prompt):
        # 使用 prompt 引导模型生成结果
        input_text = f"{prompt}\n{text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_length=1000)
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output_text

# 定义抽取信息的函数
def extract_entity_attributes(text):
    # 提供的标准化 prompt，指导 LLaMA2 输出需要的 JSON 格式实体属性
    prompt = (
        "根据以下描述，请按照如下的格式提取信息：\n"
        '输出格式为：{"犯罪行为": [], "指控罪名": [], "情节严重程度": "", "前科及初犯情况": "", "认罪情况": "", "供述情况": "", "从重处罚情况": "", "从轻处罚情况": ""}\n'
        "请在这些属性中提取出合适的信息，忽略量刑情况。\n"
    )
    
    # 初始化LLaMA信息抽取模型
    llama_extractor = LLamaExtractor()

    # 调用LLaMA生成结果
    extracted_info = llama_extractor.extract_information(text, prompt)
    
    return extracted_info

# 输入的案例描述文本
case_text = """
根据描述，孙某某和胡某某的行为构成了寻衅滋事罪。以下是相关证据支持这个结论：
在犯罪事实方面：
1. 两被告人多次阻挠罗山县“江淮南路畅通工程”施工，导致工程进度受阻，严重影响了工程进度。
2. 在施工过程中，被告人孙某某再次阻挠施工，并与施工人员发生冲突，导致孙某某头部受伤构成轻伤二级，施工人员梁某左腿腓骨骨折构成轻伤一级。
在悔过方面：
被告人胡某某在事后认识到其行为的错误，并表达了悔罪的态度，同时与受害方达成了赔偿协议，取得了受害方的谅解。
以上证据均证实了被告人孙某某、胡某某的行为已经构成了寻衅滋事罪。与犯罪事实及悔过有关的描述如下：
1. 被告人孙某某、胡某某无正当理由阻挠施工，造成二人轻伤，情节恶劣，其行为已构成寻衅滋事罪。
2. 被告人孙某某多次借故阻挠正常施工，导致二人轻伤，情节恶劣，有充分证据证明其行为构成了寻衅滋事罪。
3. 被告人胡某某到案后能如实供述自己的罪行。
4. 被告人胡某某与被害方就民事赔偿达成协议并取得被害方谅解。
"""

# 运行信息抽取并打印结果
extracted_info = extract_entity_attributes(case_text)
print(extracted_info)
