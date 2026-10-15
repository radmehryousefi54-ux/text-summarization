import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# تنظیم صفحه
# -----------------------------
st.set_page_config(
    page_title="AI Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 خلاصه‌ساز متن")
st.write("Model: `facebook/bart-large-cnn`")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------
# Input
# -----------------------------
text = st.text_area(
    "متن خود را وارد کنید:",
    height=300,
    placeholder="متن انگلیسی خود را اینجا وارد کنید..."
)


# -----------------------------
# Send Button
# -----------------------------
if st.button("🚀 ارسال", use_container_width=True):

    if not text.strip():
        st.warning("لطفاً ابتدا یک متن وارد کنید.")

    else:
        with st.spinner("در حال خلاصه‌سازی..."):

            # تبدیل متن به Token
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            # تولید خلاصه
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    min_length=30,
                    num_beams=4,
                    early_stopping=True
                )

            # تبدیل Token به متن
            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        # -----------------------------
        # نمایش جواب
        # -----------------------------
        st.subheader("📌 پاسخ مدل")

        st.info(summary)

        # -----------------------------
        # اطلاعات
        # -----------------------------
        st.divider()

        input_tokens = len(
            tokenizer.encode(text, add_special_tokens=True)
        )

        output_tokens = len(
            tokenizer.encode(summary, add_special_tokens=True)
        )

        if input_tokens > 0:
            compression = (
                1 - output_tokens / input_tokens
            ) * 100
        else:
            compression = 0

        st.subheader("📊 اطلاعات خلاصه‌سازی")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "توکن‌های ورودی",
                input_tokens
            )

        with col2:
            st.metric(
                "توکن‌های خروجی",
                output_tokens
            )

        with col3:
            st.metric(
                "فشرده‌سازی",
                f"{compression:.1f}%"
            )