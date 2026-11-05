"""Web interface: upload a photo and see whether the model thinks it is real or fake.

Run from the project folder:  python src/app.py
"""
import argparse

import gradio as gr

from predict import Detector

UNSURE_BELOW = 0.75  # confidence under this is reported as uncertain


def build_app(detector):
    def classify(image, crop_face):
        if image is None:
            return None, None, "Upload a photo to begin."

        result = detector.predict(image, crop=crop_face)
        confidence = max(result["p_real"], result["p_fake"])
        label = result["label"].upper()

        if confidence < UNSURE_BELOW:
            verdict = f"## Uncertain, leaning {label}\nConfidence is only {confidence:.0%}."
        else:
            verdict = f"## Looks {label}\nConfidence {confidence:.0%}."
        if crop_face and not result["face_found"]:
            verdict += "\n\nNo face was detected, so the whole image was used."

        scores = {"Real": result["p_real"], "Fake": result["p_fake"]}
        return scores, result["seen"], verdict

    with gr.Blocks(title="Deepfake Face Detector", analytics_enabled=False) as app:
        gr.Markdown(
            "# Deepfake Face Detector\n"
            "Upload a photo. A CNN trained on GAN and diffusion-generated images (140k Real and Fake Faces, "
            "OpenFake and others) estimates whether it is a real photo or an AI-generated fake."
        )
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Photo", sources=["upload", "clipboard"])
                crop_face = gr.Checkbox(
                    label="Crop to the face first (the model was trained on face crops)", value=False
                )
                button = gr.Button("Check photo", variant="primary")
            with gr.Column():
                verdict = gr.Markdown("Upload a photo to begin.")
                scores = gr.Label(label="Probability", num_top_classes=2)
                seen = gr.Image(label="What the model sees (resized to 224x224)", interactive=False)
        gr.Markdown(
            "**Limits:** on held-out tests it wrongly flags roughly 1 in 8 real photos as fake, and about 4 in 10 "
            "everyday scene photos. It can miss generators it has not seen, face-swap videos and edited photos. "
            "Treat the result as a hint, not proof."
        )

        outputs = [scores, seen, verdict]
        button.click(classify, [photo, crop_face], outputs)
        photo.upload(classify, [photo, crop_face], outputs)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/finetuned_final.pt")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(Detector(args.checkpoint))
    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
