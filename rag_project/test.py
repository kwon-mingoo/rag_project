from llama_cpp import Llama

MODEL = "/home/nami/Desktop/rag_project_v1/models/llama8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"  # 네 경로로 수정

llm = Llama(
    model_path=MODEL,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_batch=256,
    verbose=False,
)

messages = [
    {"role": "system", "content": "너는 한국어로 정확히 답하는 조수야."},
    {"role": "user", "content": "산업안전보건기준에 관한 규칙 제291조가 뭐에 대한 조항인지 한 문장으로 설명해줘."},
]

out = llm.create_chat_completion(
    messages=messages,
    temperature=0.2,
    top_p=0.9,
    max_tokens=256,
)

print(out["choices"][0]["message"]["content"])
