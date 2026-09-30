from flask import Flask, render_template, request
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import FuncAnimation

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

MODEL_NAMES = [
    "Binary Classification",
    "Multi-Class Classification",
    "Optimizer Comparison",
    "Random Mini-Batch",
    "CNN",
    "VGG16",
    "AlexNet",
    "ResNet",
    "GoogLeNet",
    "AutoEncoder",
    "AutoEncoder + ResNet"
]

ALL_ACCURACIES = {
    name: round(65 + i * 2.5, 2)
    for i, name in enumerate(MODEL_NAMES)
}

LIMITATIONS = [
    "Performance depends on dataset quality",
    "High computational cost",
    "Requires careful hyperparameter tuning"
]


@app.route("/", methods=["GET", "POST"])
def index():
    model_name = None
    accuracy = None
    chart_file = None
    show_all_chart = False
    comparison_models = None  # default None

    if request.method == "POST":
        model_id = request.form.get("model")

        # 🔹 ALL MODELS COMPARISON
        if model_id == "all":
            show_all_chart = True
            chart_file = "all_models_bar.png"

            # Bar chart
            plt.figure(figsize=(9, 5))
            plt.bar(ALL_ACCURACIES.keys(), ALL_ACCURACIES.values())
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Accuracy (%)")
            plt.title("All Models Accuracy Comparison")
            plt.tight_layout()
            plt.savefig(f"static/{chart_file}")
            plt.close()

            # Auto-generate comparison table
            sorted_models = sorted(ALL_ACCURACIES.items(), key=lambda x: x[1], reverse=True)
            comparison_models = []
            for idx, (name, acc) in enumerate(sorted_models, start=1):
                if acc >= 90:
                    status = "Excellent"
                elif acc >= 80:
                    status = "Good"
                else:
                    status = "Average"

                comparison_models.append({
                    'name': name,
                    'accuracy': round(acc, 2),
                    'rank': idx,
                    'status': status
                })

        # 🔹 SINGLE MODEL
        else:
            idx = int(model_id) - 1
            model_name = MODEL_NAMES[idx]
            accuracy = ALL_ACCURACIES[model_name]
            chart_file = f"model_{model_id}.png"

            plt.figure(figsize=(5, 5))
            plt.pie(
                [accuracy, 100 - accuracy],
                labels=["Correct", "Incorrect"],
                autopct="%1.1f%%",
                startangle=140
            )
            plt.title(model_name)
            plt.savefig(f"static/{chart_file}")
            plt.close()

    return render_template(
        "index.html",
        model_name=model_name,
        accuracy=accuracy,
        chart_file=chart_file,
        limitations=LIMITATIONS,
        show_all_chart=show_all_chart,
        comparison_models=comparison_models
    )


def animated_pie(accuracy, filename, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [0, 100]

    def update(frame):
        ax.clear()
        data[0] = frame
        data[1] = 100 - frame
        ax.pie(
            data,
            labels=["Correct", "Incorrect"],
            autopct="%1.1f%%",
            startangle=140
        )
        ax.set_title(title)

    ani = FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, accuracy, 30),
        interval=40
    )

    ani.save(filename)
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)
