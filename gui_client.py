import tkinter as tk
from tkinter import ttk, messagebox
import requests
import threading

last_input_data = None


def add_label_entry(parent, text, row):
    tk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
    entry = tk.Entry(parent, width=20)
    entry.grid(row=row, column=1, padx=5, pady=5)
    return entry


def add_dropdown(parent, text, options, row):
    tk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
    var = tk.StringVar(value="")
    box = ttk.Combobox(
        parent,
        textvariable=var,
        values=options,
        state="readonly",
        width=17
    )
    box.grid(row=row, column=1, padx=5, pady=5)
    return var


def predict():
    global last_input_data

    try:
        required_dropdowns = {
            "Internet access": internet_var.get(),
            "Extracurricular activities": extra_var.get(),
            "Part-time job": job_var.get(),
            "Parent support": support_var.get(),
            "Romantic relationship": romantic_var.get(),
            "Gender": gender_var.get(),
            "Race": race_var.get(),
            "School type": school_var.get(),
            "Parental education": education_var.get(),
            "Locale": locale_var.get()
        }

        for field_name, value in required_dropdowns.items():
            if not value:
                messagebox.showerror(
                    "Missing Input",
                    f"Please select a value for {field_name}."
                )
                return

        attendance_percent = float(attendance_entry.get())

        data = {
            "age": int(age_entry.get()),
            "grade": int(grade_entry.get()),
            "ses_quartile": int(ses_entry.get()),
            "math_score": float(math_entry.get()),
            "reading_score": float(reading_entry.get()),
            "science_score": float(science_entry.get()),
            "attendance_rate": attendance_percent / 100.0,
            "study_hours": float(study_entry.get()),
            "internet_access": internet_var.get() == "Yes",
            "extracurricular": extra_var.get() == "Yes",
            "part_time_job": job_var.get() == "Yes",
            "parent_support": support_var.get() == "Yes",
            "romantic": romantic_var.get() == "Yes",
            "free_time": int(free_time_entry.get()),
            "go_out": int(go_out_entry.get()),
            "gender": gender_var.get(),
            "race": race_var.get(),
            "school_type": school_var.get(),
            "parental_education": education_var.get(),
            "locale": locale_var.get()
        }

        last_input_data = data.copy()

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            low = result["probabilities"]["low"] * 100
            medium = result["probabilities"]["medium"] * 100
            high = result["probabilities"]["high"] * 100
            category = result["predicted_category"].upper()

            if result["predicted_category"] == "high":
                interpretation = "The student is highly likely to fall in the high GPA category."
            elif result["predicted_category"] == "medium":
                interpretation = "The student is most likely to fall in the medium GPA category."
            else:
                interpretation = "The student is most likely to fall in the low GPA category."

            prediction = (
                "\n" + "-" * 40 + "\n"
                f"Prediction #{predict.counter}\n"
                + "-" * 40 + "\n"
                f"Predicted GPA Category: {category}\n\n"
                f"Confidence Levels\n"
                f"Low GPA    : {low:.2f}%\n"
                f"Medium GPA : {medium:.2f}%\n"
                f"High GPA   : {high:.2f}%\n\n"
                f"Interpretation\n{interpretation}\n"
            )

            result_log.insert(tk.END, prediction)
            result_log.see(tk.END)
            predict.counter += 1

        else:
            result_log.insert(
                tk.END,
                f"\nError: {response.json().get('error', 'Unknown error')}\n"
            )
            result_log.see(tk.END)

    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error",
            "Make sure app.py is running."
        )
    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numeric values."
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def submit_feedback_thread():
    thread = threading.Thread(target=submit_feedback, daemon=True)
    thread.start()


def submit_feedback():
    global last_input_data

    try:
        if last_input_data is None:
            messagebox.showerror(
                "Feedback Error",
                "Please make a prediction first."
            )
            return

        true_label = feedback_var.get()

        if not true_label:
            messagebox.showerror(
                "Feedback Error",
                "Please select the correct GPA category."
            )
            return

        feedback = last_input_data.copy()
        feedback["true_label"] = true_label

        response = requests.post(
            "http://127.0.0.1:5000/feedback",
            json=feedback
        )

        if response.status_code == 200:
            response_data = response.json()

            message = "Feedback saved successfully."
            if response_data.get("retrained"):
                message += "\nThe model was retrained."

            messagebox.showinfo("Feedback Saved", message)
            feedback_var.set("")
        else:
            messagebox.showerror(
                "Feedback Error",
                response.json().get("error", "Unknown error")
            )

    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error",
            "Make sure app.py is running."
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Student Performance Prediction App")
root.geometry("1000x700")

predict.counter = 1

title = tk.Label(
    root,
    text="Student Performance Prediction App",
    font=("Arial", 18, "bold")
)
title.pack(pady=10)

main = tk.Frame(root)
main.pack(fill="both", expand=True, padx=20, pady=10)

left_side = tk.LabelFrame(
    main,
    text="Student Input",
    padx=10,
    pady=10
)
left_side.pack(side="left", fill="both", expand=True, padx=10, pady=10)

right_side = tk.LabelFrame(
    main,
    text="Prediction Output",
    padx=15,
    pady=15
)
right_side.pack(side="right", fill="both", expand=True, padx=10, pady=10)

age_entry = add_label_entry(left_side, "Age", 0)
grade_entry = add_label_entry(left_side, "Grade", 1)
ses_entry = add_label_entry(left_side, "SES Quartile (1-4)", 2)
math_entry = add_label_entry(left_side, "Math score", 3)
reading_entry = add_label_entry(left_side, "Reading score", 4)
science_entry = add_label_entry(left_side, "Science score", 5)
attendance_entry = add_label_entry(left_side, "Attendance rate (%)", 6)
study_entry = add_label_entry(left_side, "StudyHours value (per day)", 7)
free_time_entry = add_label_entry(left_side, "Free time (Scale of 1-5)", 8)
go_out_entry = add_label_entry(left_side, "Go out frequency (Scale of 1-5)", 9)

internet_var = add_dropdown(left_side, "Internet access", ["Yes", "No"], 10)
extra_var = add_dropdown(left_side, "Extracurricular", ["Yes", "No"], 11)
job_var = add_dropdown(left_side, "Part-time job", ["Yes", "No"], 12)
support_var = add_dropdown(left_side, "Parent support", ["Yes", "No"], 13)
romantic_var = add_dropdown(left_side, "Romantic relationship", ["Yes", "No"], 14)

gender_var = add_dropdown(left_side, "Gender", ["Female", "Male"], 15)
race_var = add_dropdown(
    left_side,
    "Race",
    ["Asian", "Black", "Hispanic", "Other", "Two-or-more", "White"],
    16
)
school_var = add_dropdown(left_side, "School type", ["Private", "Public"], 17)
education_var = add_dropdown(
    left_side,
    "Parental education",
    ["<HS", "HS", "Bachelors+", "SomeCollege"],
    18
)
locale_var = add_dropdown(
    left_side,
    "Locale",
    ["City", "Rural", "Suburban", "Town"],
    19
)

predict_button = tk.Button(
    left_side,
    text="Predict",
    command=predict,
    font=("Arial", 12, "bold"),
    width=20
)
predict_button.grid(row=20, column=0, columnspan=2, pady=15)

log = tk.Frame(right_side)
log.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(log)
scrollbar.pack(side="right", fill="y")

result_log = tk.Text(
    log,
    wrap="word",
    font=("Arial", 11),
    yscrollcommand=scrollbar.set
)
result_log.pack(side="left", fill="both", expand=True)

scrollbar.config(command=result_log.yview)

feedback_frame = tk.LabelFrame(
    right_side,
    text="Submit Feedback",
    padx=10,
    pady=10
)
feedback_frame.pack(fill="x", padx=5, pady=10)

feedback_var = tk.StringVar(value="")

tk.Label(
    feedback_frame,
    text="Correct GPA Category:"
).grid(row=0, column=0, padx=5, pady=5, sticky="w")

feedback_box = ttk.Combobox(
    feedback_frame,
    textvariable=feedback_var,
    values=["low", "medium", "high"],
    state="readonly",
    width=15
)
feedback_box.grid(row=0, column=1, padx=5, pady=5)

feedback_button = tk.Button(
    feedback_frame,
    text="Submit Feedback",
    command=submit_feedback_thread,
    font=("Arial", 10, "bold")
)
feedback_button.grid(row=1, column=0, columnspan=2, pady=10)

root.mainloop()
