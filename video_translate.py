def run_streamlit_app():
    if st is None:
        raise RuntimeError('streamlit not installed. Run: pip install streamlit')

    st.set_page_config(page_title="Rayvia AI – Video Translator", layout="wide")

    st.markdown("""
    <h2 style="text-align:center; font-size:32px; font-weight:700;">
        🎬 Rayvia AI – Video Translator & Dubbing Tool
    </h2>
    <p style="text-align:center; font-size:16px;">
        Upload a video → Select languages → Translate → Download<br>
        Supports long videos, all languages, subtitles & dubbing, HD output.
    </p>
    <hr>
    """, unsafe_allow_html=True)

    st.subheader("📤 Upload Video File")
    uploaded = st.file_uploader(
        'Upload video file',
        type=['mp4', 'mov', 'mkv', 'avi', 'mpeg4']
    )

    src_lang = st.text_input("Source language code (e.g. en, auto)", value="en")
    tgt_lang = st.text_input("Target language code (e.g. hi, ur, ar, fr)", value="hi")

    mode = st.selectbox("Mode", ['subtitles', 'dubbed'])
    whisper_model = st.selectbox("Whisper model", ['tiny', 'base', 'small', 'medium', 'large'], index=2)

    if uploaded:
        tmpdir = tempfile.mkdtemp(prefix='rayvia_upload_')
        in_path = os.path.join(tmpdir, uploaded.name)

        with open(in_path, 'wb') as f:
            f.write(uploaded.getbuffer())

        st.success(f"Video uploaded: {uploaded.name}")

        if st.button("🚀 Start Translation"):
            with st.spinner("Processing… This may take time for long videos…"):
                args = argparse.Namespace(
                    input=in_path,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    out=os.path.join(tmpdir, f'output_{tgt_lang}_{uploaded.name}'),
                    mode=mode,
                    whisper_model=whisper_model
                )
                try:
                    run_pipeline(args)
                    output_file = args.out

                    if os.path.exists(output_file):
                        st.success("🎉 Translation complete!")
                        with open(output_file, "rb") as f:
                            st.download_button(
                                "Download Translated Video",
                                f,
                                file_name=os.path.basename(output_file)
                            )
                    else:
                        st.error("Output file not found. Something went wrong.")

                except Exception as e:
                    st.error(f"Error: {e}")
