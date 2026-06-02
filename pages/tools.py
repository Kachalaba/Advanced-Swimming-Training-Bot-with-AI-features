"""
Video tools page module.
"""

from pathlib import Path

import streamlit as st

from video_analysis.video_tools import (
    create_side_by_side,
    create_tracked_zoom,
    create_zoom_video,
    extract_highlight,
    get_video_info,
)


def render_tools_tab():
    """Render video tools tab."""

    st.markdown('<div class="section-title">🎬 Відео інструменти</div>', unsafe_allow_html=True)

    tool_tab1, tool_tab2, tool_tab3 = st.tabs(["⚖️ Side-by-Side", "✂️ Highlight", "🔍 Zoom"])

    # ========================================================================
    # SIDE-BY-SIDE
    # ========================================================================
    with tool_tab1:
        st.markdown("### ⚖️ Порівняння відео Side-by-Side")
        st.markdown("Завантажте два відео для порівняння техніки")

        col1, col2 = st.columns(2)

        with col1:
            video1 = st.file_uploader("📹 Відео 1", type=["mp4", "mov", "avi"], key="sbs_video1")
            label1 = st.text_input("Підпис 1", value="До", key="sbs_label1")

        with col2:
            video2 = st.file_uploader("📹 Відео 2", type=["mp4", "mov", "avi"], key="sbs_video2")
            label2 = st.text_input("Підпис 2", value="Після", key="sbs_label2")

        if video1 and video2:
            if st.button("🎬 Створити порівняння", type="primary", use_container_width=True):
                with st.spinner("Створюємо відео..."):
                    output_dir = Path("streamlit_outputs/video_tools")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save uploaded files
                    v1_path = output_dir / f"temp_v1_{video1.name}"
                    v2_path = output_dir / f"temp_v2_{video2.name}"

                    with open(v1_path, "wb") as f:
                        f.write(video1.read())
                    with open(v2_path, "wb") as f:
                        f.write(video2.read())

                    output_path = output_dir / "side_by_side.mp4"
                    result = create_side_by_side(
                        str(v1_path),
                        str(v2_path),
                        str(output_path),
                        labels=(label1, label2),
                    )

                    if result:
                        st.success("✅ Відео створено!")
                        st.video(str(output_path))

                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Завантажити",
                                f,
                                file_name="comparison.mp4",
                                mime="video/mp4",
                            )
                    else:
                        st.error("❌ Помилка створення відео")

    # ========================================================================
    # HIGHLIGHT EXTRACTION
    # ========================================================================
    with tool_tab2:
        st.markdown("### ✂️ Вирізати фрагмент")
        st.markdown("Виріжте важливий момент з відео")

        video_hl = st.file_uploader("📹 Відео", type=["mp4", "mov", "avi"], key="hl_video")

        if video_hl:
            output_dir = Path("streamlit_outputs/video_tools")
            output_dir.mkdir(parents=True, exist_ok=True)

            temp_path = output_dir / f"temp_{video_hl.name}"
            with open(temp_path, "wb") as f:
                f.write(video_hl.read())

            info = get_video_info(str(temp_path))
            if info:
                st.info(f"📊 Відео: {info.duration_sec:.1f}с, {info.fps:.0f} FPS, {info.width}x{info.height}")

                col1, col2 = st.columns(2)
                with col1:
                    start_sec = st.number_input(
                        "⏱️ Початок (сек)",
                        min_value=0.0,
                        max_value=info.duration_sec,
                        value=0.0,
                        step=0.5,
                    )
                with col2:
                    end_sec = st.number_input(
                        "⏱️ Кінець (сек)",
                        min_value=0.0,
                        max_value=info.duration_sec,
                        value=min(5.0, info.duration_sec),
                        step=0.5,
                    )

                col1, col2 = st.columns(2)
                with col1:
                    slow_factor = st.select_slider(
                        "🐢 Швидкість",
                        options=[0.25, 0.5, 0.75, 1.0],
                        value=1.0,
                        format_func=lambda x: f"{x}x",
                    )
                with col2:
                    add_text = st.text_input("📝 Текст на відео", value="", key="hl_text")

                if st.button("✂️ Вирізати", type="primary", use_container_width=True):
                    with st.spinner("Вирізаємо фрагмент..."):
                        output_path = output_dir / "highlight.mp4"
                        result = extract_highlight(
                            str(temp_path),
                            str(output_path),
                            start_sec,
                            end_sec,
                            add_text=add_text if add_text else None,
                            slow_factor=slow_factor,
                        )

                        if result:
                            st.success(f"✅ Фрагмент вирізано ({end_sec - start_sec:.1f}с)")
                            st.video(str(output_path))

                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "📥 Завантажити",
                                    f,
                                    file_name="highlight.mp4",
                                    mime="video/mp4",
                                )

    # ========================================================================
    # ZOOM
    # ========================================================================
    with tool_tab3:
        st.markdown("### 🔍 Zoom відео")
        st.markdown("Збільште частину відео для детального аналізу")

        video_zoom = st.file_uploader("📹 Відео", type=["mp4", "mov", "avi"], key="zoom_video")

        if video_zoom:
            output_dir = Path("streamlit_outputs/video_tools")
            output_dir.mkdir(parents=True, exist_ok=True)

            temp_path = output_dir / f"temp_zoom_{video_zoom.name}"
            with open(temp_path, "wb") as f:
                f.write(video_zoom.read())

            info = get_video_info(str(temp_path))
            if info:
                st.info(f"📊 Відео: {info.width}x{info.height}")

                zoom_type = st.radio(
                    "Тип zoom",
                    ["📍 Фіксована область", "🎯 Трекінг об'єкта"],
                    horizontal=True,
                )

                zoom_factor = st.slider("🔍 Zoom", min_value=1.5, max_value=4.0, value=2.0, step=0.5)

                if zoom_type == "📍 Фіксована область":
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.slider("X позиція", 0, info.width, info.width // 2)
                        w = st.slider("Ширина", 100, info.width // 2, info.width // 4)
                    with col2:
                        y = st.slider("Y позиція", 0, info.height, info.height // 2)
                        h = st.slider("Висота", 100, info.height // 2, info.height // 4)

                    if st.button("🔍 Створити Zoom", type="primary", use_container_width=True):
                        with st.spinner("Створюємо zoom відео..."):
                            output_path = output_dir / "zoomed.mp4"
                            result = create_zoom_video(
                                str(temp_path),
                                str(output_path),
                                region=(x, y, w, h),
                                zoom_factor=zoom_factor,
                            )

                            if result:
                                st.success("✅ Zoom відео створено!")
                                st.video(str(output_path))

                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "📥 Завантажити",
                                        f,
                                        file_name="zoomed.mp4",
                                        mime="video/mp4",
                                    )
                else:
                    st.info("🎯 Трекінг автоматично слідкує за виявленим спортсменом. Спочатку проведіть аналіз відео.")

                    if st.button(
                        "🔍 Створити Tracking Zoom",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner("Виявляємо та zoom..."):
                            # Quick detection for tracking
                            from video_analysis.frame_extractor import extract_frames_from_video
                            from video_analysis.swimmer_detector import detect_swimmer_in_frames

                            frames_dir = output_dir / "temp_frames"
                            frame_result = extract_frames_from_video(str(temp_path), str(frames_dir), fps=10)
                            detection_result = detect_swimmer_in_frames(frame_result["frames"], str(frames_dir))

                            output_path = output_dir / "tracked_zoom.mp4"
                            result = create_tracked_zoom(
                                str(temp_path),
                                str(output_path),
                                detection_result["detections"],
                                zoom_factor=zoom_factor,
                            )

                            if result:
                                st.success("✅ Tracking zoom створено!")
                                st.video(str(output_path))

                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "📥 Завантажити",
                                        f,
                                        file_name="tracked_zoom.mp4",
                                        mime="video/mp4",
                                    )
