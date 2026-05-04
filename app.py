import importlib
from collections.abc import Iterator
from typing import Protocol, cast

import streamlit as st
from google import genai
from google.genai import types


class StreamlitJsEval(Protocol):
    def __call__(self, *, js_expressions: str) -> object: ...


_streamlit_js_eval_mod = importlib.import_module("streamlit_js_eval")
streamlit_js_eval: StreamlitJsEval = cast(
    StreamlitJsEval,
    getattr(_streamlit_js_eval_mod, "streamlit_js_eval"),
)

# Setting up the Streamlit page configuration
st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="💬")
_ = st.title("Job Interview Assistant")

# Initialize session state variables
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False
if "chat_complete" not in st.session_state:
    st.session_state.chat_complete = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# Helper functions to update session state
def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True

# Setup stage for collecting user details
if not st.session_state.setup_complete:
    _ = st.subheader('Personal Information')

    if "name" not in st.session_state:
        st.session_state["name"] = ""
    if "experience" not in st.session_state:
        st.session_state["experience"] = ""
    if "skills" not in st.session_state:
        st.session_state["skills"] = ""

    st.session_state["name"] = st.text_input(
        label="Name",
        value=cast(str, st.session_state["name"]),
        placeholder="Enter your name",
        max_chars=40,
    )
    st.session_state["experience"] = st.text_area(
        label="Experience",
        value=cast(str, st.session_state["experience"]),
        placeholder="Describe your experience",
        max_chars=200,
    )
    st.session_state["skills"] = st.text_area(
        label="Skills",
        value=cast(str, st.session_state["skills"]),
        placeholder="List your skills",
        max_chars=200,
    )

    _ = st.subheader('Company and Position')

    if "level" not in st.session_state:
        st.session_state["level"] = "Junior"
    if "position" not in st.session_state:
        st.session_state["position"] = "Data Scientist"
    if "company" not in st.session_state:
        st.session_state["company"] = "Amazon"

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["level"] = st.radio(
            "Choose level",
            key="visibility",
            options=["Junior", "Mid-level", "Senior"],
            index=["Junior", "Mid-level", "Senior"].index(cast(str, st.session_state["level"]))
        )

    with col2:
        st.session_state["position"] = st.selectbox(
            "Choose a position",
            ("Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"),
            index=(
                "Data Scientist",
                "Data Engineer",
                "ML Engineer",
                "BI Analyst",
                "Financial Analyst",
            ).index(cast(str, st.session_state["position"]))
        )

    st.session_state["company"] = st.selectbox(
        "Select a Company",
        ("Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"),
        index=("Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify").index(
            cast(str, st.session_state["company"])
        )
    )

    if st.button("Start Interview", on_click=complete_setup):
        st.write("Setup complete. Starting interview...")

# Interview phase
if st.session_state.setup_complete and not st.session_state.feedback_shown and not st.session_state.chat_complete:

    _ = st.info(
        """
        Start by introducing yourself
        """,
        icon="👋",
    )

    # Initialize Gemini client
    client = genai.Client(api_key=cast(str, st.secrets["GEMINI_API_KEY"]))

    if "gemini_model" not in st.session_state:
        st.session_state["gemini_model"] = cast(str, st.secrets["GEMINI_MODEL"])

    # System prompt for the interview
    SYSTEM_PROMPT = (
        f"You are an HR executive that interviews an interviewee called {st.session_state['name']} "
        f"with experience {st.session_state['experience']} and skills {st.session_state['skills']}. "
        f"You should interview him for the position {st.session_state['level']} {st.session_state['position']} "
        f"at the company {st.session_state['company']}"
    )

    messages = cast(list[dict[str, str]], st.session_state.messages)

    # Display chat messages
    for message in messages:
        with st.chat_message(message["role"]):
            _ = st.markdown(message["content"])

    # Handle user input and Gemini response
    if st.session_state.user_message_count < 5:
        if prompt := st.chat_input("Your response", max_chars=1000):
            _ = messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                _ = st.markdown(prompt)

            if st.session_state.user_message_count < 4:
                # Build conversation history in Gemini format
                history = [
                    types.Content(
                        role=m["role"],
                        parts=[types.Part(text=m["content"])],
                    )
                    for m in messages[:-1]
                ]

                with st.chat_message("assistant"):
                    def stream_response():
                        for chunk in client.models.generate_content_stream(
                            model=cast(str, st.session_state["gemini_model"]),
                            contents=history
                            + [
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=prompt)],
                                ),
                            ],  # pyright: ignore[reportArgumentType]
                            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                        ):
                            if chunk.text:
                                yield chunk.text

                    response = st.write_stream(stream_response())
                _ = messages.append({"role": "assistant", "content": str(response)})

            st.session_state.user_message_count += 1

    if st.session_state.user_message_count >= 5:
        st.session_state.chat_complete = True


# Interview finished — offer feedback (matches app-final flow; next rerun skips chat UI)
if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button("Get Feedback", on_click=show_feedback):
        _ = st.write("Fetching feedback...")


FEEDBACK_SYSTEM = """You are a helpful tool that provides feedback on an interviewee performance.
Before the Feedback give a score of 1 to 10.
Follow this format:
Overall Score: //Your score
Feedback: //Here you put your feedback
Give only the feedback do not ask any additional questions.
"""


# Feedback screen
if st.session_state.feedback_shown:
    _ = st.subheader("Feedback")

    messages_fb = cast(list[dict[str, str]], st.session_state.messages)
    conversation_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages_fb]
    )

    feedback_client = genai.Client(api_key=cast(str, st.secrets["GEMINI_API_KEY"]))
    model_id = cast(
        str,
        st.session_state["gemini_model"]
        if "gemini_model" in st.session_state
        else st.secrets["GEMINI_MODEL"],
    )

    user_blob = (
        "This is the interview you need to evaluate. "
        "You are only a tool for scoring and feedback — do not start a conversation.\n\n"
        f"{conversation_history}"
    )

    if not st.session_state.get("feedback_body"):

        def stream_feedback() -> Iterator[str]:
            fb_stream = feedback_client.models.generate_content_stream(
                model=model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_blob)],
                    ),
                ],
                config=types.GenerateContentConfig(system_instruction=FEEDBACK_SYSTEM),
            )
            for chunk in fb_stream:
                if chunk.text:
                    yield chunk.text

        with st.spinner("Generating feedback…"):
            streamed = str(st.write_stream(stream_feedback()))
        st.session_state.feedback_body = streamed
    else:
        _ = st.markdown(cast(str, st.session_state["feedback_body"]))

    if st.button("Restart Interview", type="primary"):
        _ = streamlit_js_eval(js_expressions="parent.window.location.reload()")