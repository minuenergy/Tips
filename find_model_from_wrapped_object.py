# find exact model ( nn.module's ) from complicated wrapped Objects ( ex : huggingface, vllm .. )
import torch.nn as nn

def find_model_recursively(obj, max_depth=10, current_depth=0, path=[]):
    if current_depth > max_depth:
        return None, None

    # 기본적인 모델 타입 체크
    if isinstance(obj, nn.Module):
        return obj, path

    # 객체의 속성들을 탐색
    for attr_name in dir(obj):
        if attr_name.startswith('__'):  # 특수 메서드는 건너뜁니다
            continue

        try:
            attr = getattr(obj, attr_name)
        except Exception:  # 일부 속성에 접근할 때 에러가 발생할 수 있습니다
            continue

        # 재귀적으로 속성 탐색
        model, sub_path = find_model_recursively(attr, max_depth, current_depth + 1, path + [attr_name])
        if model is not None:
            return model, sub_path

    return None, None

def extract_model_from_llm(llm):
    model, path = find_model_recursively(llm)
    if model is not None:
        print(f"Found model at path: {' -> '.join(path)}")
        return model
    else:
        print("Could not find the model. Please inspect the LLM object structure manually.")
        return None

# 사용 예시
llm = LLM(model="llava-hf/llava-1.5-7b-hf")
model = extract_model_from_llm(llm)
if model:
    print(type(model))
    print(model)
