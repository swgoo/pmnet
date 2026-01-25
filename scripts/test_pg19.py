import torch
import math
from tqdm import tqdm
from datasets import load_dataset

from pmnet.modeling_pmnet import PMNetForCausalLM
from pmnet.tokenization_pmnet import ByteTokenizer

# -----------------------------------------------------------------------------
# 설정 (본인 환경에 맞게 수정하세요)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 8192  # 모델 학습 시 사용한 Context Length (또는 Sliding Window 크기보다 크게)
NUM_BOOKS = 10  # 평가할 책의 권수 (전체 다 하면 오래 걸리니 10~50권 정도 추천)


def evaluate_pg19(model, tokenizer):
    print(f"Loading PG-19 dataset (test split)...")
    # streaming=True를 사용하여 거대한 데이터셋을 다운로드 없이 바로 읽습니다.
    dataset = load_dataset("emozilla/pg19", split="test", streaming=True)

    model.eval()
    model.to(DEVICE)

    total_nll = 0.0  # Total Negative Log Likelihood (Loss 합)
    total_tokens = 0  # 총 토큰 수

    print(f"Starting evaluation on {NUM_BOOKS} books...")

    # 데이터셋 순회
    for book_idx, sample in enumerate(dataset):
        if book_idx >= NUM_BOOKS:
            break

        text = sample["text"]

        # 1. 토큰화 (전체 책을 한 번에 인코딩)
        # 주의: 메모리가 부족하면 tokenizer도 청크 단위로 해야 하지만, 보통 텍스트는 램에 들어갑니다.
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(DEVICE)  # Shape: [1, Total_Book_Len]

        seq_len_book = input_ids.size(1)
        print(f"Book {book_idx+1}: {seq_len_book} tokens processing...")

        # 2. State 초기화 (책이 바뀔 때마다 기억을 리셋!)
        # PMNet은 forward에서 past_key_values가 None이면 새로 생성하므로 None으로 시작
        past_key_values = None

        book_nll = 0.0
        book_tokens = 0

        # 3. 청크 단위로 잘라서 순전파 (State Passing)
        # stride 없이 SEQ_LEN만큼 딱딱 잘라서 넣습니다.
        with torch.no_grad():
            for i in tqdm(
                range(0, seq_len_book, SEQ_LEN), desc=f"Book {book_idx+1}", leave=False
            ):
                # 마지막 자투리가 너무 짧으면 건너뛰거나 패딩할 수 있음 (여기선 포함)
                end_i = min(i + SEQ_LEN, seq_len_book)
                chunk = input_ids[:, i:end_i]

                # 타겟(Labels)은 입력과 동일 (모델 내부에서 shift함)
                # use_cache=True를 켜야 past_key_values(Memory State)가 반환됨
                outputs = model(
                    input_ids=chunk,
                    labels=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Loss 계산 (CrossEntropyLoss는 기본적으로 평균(mean)이므로 토큰 수를 곱해줌)
                # HuggingFace 모델 출력의 loss는 이미 shift가 고려된 loss임
                loss = outputs.loss

                # 다음 청크를 위해 State 업데이트 (State Passing의 핵심!)
                past_key_values = outputs.past_key_values

                # 유효 토큰 수 계산 (마지막 청크 등 고려)
                # 보통 첫 토큰은 예측 못하므로 loss 계산에서 제외되지만, 근사를 위해 청크 길이 사용
                chunk_len = chunk.size(1)

                book_nll += loss.item() * chunk_len
                book_tokens += chunk_len

        # 책 한 권 끝났을 때 BPB 중간 집계
        if book_tokens > 0:
            book_bpb = (book_nll / book_tokens) / math.log(2)
            print(f" -> Book {book_idx+1} BPB: {book_bpb:.4f}")

        total_nll += book_nll
        total_tokens += book_tokens

    # 4. 최종 결과 출력
    final_bpb = (total_nll / total_tokens) / math.log(2)
    print("=" * 40)
    print(f"Final PG-19 BPB (over {total_tokens} tokens): {final_bpb:.4f}")
    print("=" * 40)

    return final_bpb


# -----------------------------------------------------------------------------
# 실행부 (사용자 환경)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 모델과 토크나이저 로드 (사용자님의 로드 방식에 맞게 수정)
    # 예: 로컬 폴더에서 불러오거나, 커스텀 클래스 인스턴스화
    model = PMNetForCausalLM.from_pretrained("ckpts/byte_batch48_28000").to(DEVICE)
    tokenizer = ByteTokenizer()

    # [가정] 이미 model과 tokenizer가 메모리에 로드되어 있다고 가정
    # model = my_pmnet_model
    # tokenizer = my_byte_tokenizer

    # 평가 실행
    evaluate_pg19(model, tokenizer)
