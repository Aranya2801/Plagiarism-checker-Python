from _typeshed import Self
from msilib.schema import SelfReg
import os
import string
import tkinter as tk
import tkinter.filedialog as filedialog
from difflib import SequenceMatcher

# Default similarity threshold
DEFAULT_THRESHOLD = 0.6


def preprocess_text(text):
    """
    Preprocesses the input text by removing any non-alphanumeric characters and converting to lowercase.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    The preprocessed text.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text


def plagiarism_check(file1, file2, threshold=DEFAULT_THRESHOLD, preprocess=True):
    """
    Checks if two files are similar by computing the ratio of matching text.

    Parameters:
    file1 (str): The path to the first file to compare.
    file2 (str): The path to the second file to compare.
    threshold (float): The similarity threshold above which plagiarism is detected. Default is 0.6.
    preprocess (bool): Whether to preprocess the text before comparing. Default is True.

    Returns:
    A tuple containing the similarity ratio and the matching text.
    """
    if not os.path.isfile(file1):
        print(f"Error: {file1} does not exist.")
        return None, None
    if not os.path.isfile(file2):
        print(f"Error: {file2} does not exist.")
        return None, None

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        text1 = f1.read()
        text2 = f2.read()

    if preprocess:
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

    similarity_ratio = SequenceMatcher(None, text1, text2).ratio()

    if similarity_ratio >= threshold:
        return similarity_ratio, text1

    return similarity_ratio, None


def batch_check(directory, threshold=DEFAULT_THRESHOLD, preprocess=True):
    """
    Checks for plagiarism between all pairs of text files in a directory.

    Parameters:
    directory (str): The path to the directory containing the files.
    threshold (float): The similarity threshold above which plagiarism is detected. Default is 0.6.
    preprocess (bool): Whether to preprocess the text before comparing. Default is True.

    Returns:
    A list of tuples containing the file names and similarity ratios for all pairs of files above the threshold.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} does not exist.")
        return None

    def batch_check(directory, threshold=DEFAULT_THRESHOLD, preprocess=True, output_file=None):
# Perform the plagiarism check on all pairs of files in the directory
                similarities = []
    files = os.listdir(directory)
    for i in range(len(files)):
        file1 = files[i]
        for j in range(i+1, len(files)):
            file2 = files[j]
            similarity_ratio, matching_text = plagiarism_check(os.path.join(
                directory, file1), os.path.join(directory, file2), threshold=threshold, preprocess=preprocess)
            if matching_text:
                similarities.append(
                    (file1, file2, similarity_ratio, matching_text))
class PlagiarismCheckerGUI:
            def init(self):
                    self.window = tk.Tk()
                    self.window.title("Plagiarism Checker")
    # Create the file selection buttons
                    self.file1_button = tk.Button(self.window, text="Select File 1", command=self.select_file1)
                    self.file1_button.grid(row=0, column=0)

                    self.file2_button = tk.Button(self.window, text="Select File 2", command=self.select_file2)
                    self.file2_button.grid(row=1, column=0)

    # Create the threshold input
                    threshold_var = tk.StringVar()
                    threshold_var.set(str(DEFAULT_THRESHOLD))

    # Create the preprocess checkbox
preprocess_var = tk.BooleanVar()
preprocess_var.set(True)

preprocess_label = tk.Label(window, text="Preprocess:")
preprocess_label.grid(row=2, column=0)

preprocess_checkbox = tk.Checkbutton(
    window, variable=preprocess_var, onvalue=True, offvalue=False)
preprocess_checkbox.grid(row=2, column=1)

def select_file1(self):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        self.file1_button.config(text=file_path)

def select_file2(self):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        self.file2_button.config(text=file_path)

def check_plagiarism(self):
    file1 = self.file1_button.cget("text")
    file2 = self.file2_button.cget("text")
    threshold = float(self.threshold_entry.get())
    preprocess = self.preprocess_var.get()

    if file1 and file2:
        similarity_ratio, matching_text = plagiarism_check(file1, file2, threshold=threshold, preprocess=preprocess)

        if matching_text:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Similarity Ratio: {similarity_ratio:.2f}\n\n")
            self.results_text.insert(tk.END, f"Matching Text:\n\n{matching_text}")
        else:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No plagiarism detected.")
    else:
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Please select two files to compare.")
    # Initialize the GUI
    gui = PlagiarismCheckerGUI()

    # Run the GUI main loop
    gui.window.mainloop()
if name == 'main':
    # Example usage:
        similarities = batch_check("texts")
for file1, file2, similarity_ratio, matching_text in similarities:
        print(f"{file1} is {similarity_ratio:.2f}% similar to {file2}\nMatching Text:\n{matching_text}\n")
# Run the GUI
run_gui()
import csv

def batch_check(directory, threshold=DEFAULT_THRESHOLD, preprocess=True, output_file=None):
    # Save the results to a CSV file if an output file is specified
    if output_file is not None:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File 1', 'File 2', 'Similarity Ratio', 'Matching Text'])
            for file1, file2, similarity_ratio, matching_text in similarities:
                writer.writerow([file1, file2, similarity_ratio, matching_text])

    return similarities

def run_gui():
    def select_output_file():
        output_file = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV Files", "*.csv")])
        if output_file:
            output_file_var.set(output_file)

    def batch_check_wrapper():
        # Get the input parameters from the GUI
        directory = directory_var.get()
        threshold = float(threshold_var.get())
        preprocess = preprocess_var.get()
        output_file = output_file_var.get()

        # Perform the batch check
        similarities = batch_check(directory, threshold=threshold, preprocess=preprocess, output_file=output_file)

        # Display the results in the GUI
        if output_file is None:
            results_text.delete(1.0, tk.END)
            for file1, file2, similarity_ratio, matching_text in similarities:
                results_text.insert(tk.END, f"{file1} is {similarity_ratio:.2f}% similar to {file2}\nMatching Text:\n{matching_text}\n\n")
            messagebox.showinfo(title="Plagiarism Checker", message="Batch check complete.")
        else:
            messagebox.showinfo(title="Plagiarism Checker", message=f"Batch check complete.\nResults saved to {output_file}.")

    # Initialize the GUI
    window = tk.Tk()
    window.title("Plagiarism Checker")

    # Create the directory selection button
    directory_var = tk.StringVar()
    directory_var.set(os.getcwd())

    directory_label = tk.Label(window, text="Directory:")
    directory_label.grid(row=0, column=0)

    directory_entry = tk.Entry(window, textvariable=directory_var)
    directory_entry.grid(row=0, column=1)

    directory_button = tk.Button(window, text="Select Directory", command=lambda: directory_var.set(filedialog.askdirectory()))
    directory_button.grid(row=0, column=2)

# Create the output file selection button
output_file_var = tk.StringVar()

output_file_label = tk.Label(window, text="Output File:")
output_file_label.grid(row=1, column=0)

output_file_entry = tk.Entry(window, textvariable=output_file_var)
output_file_entry.grid(row=1, column=1)

output_file_button = tk.Button(window, text="Select Output File", command=select_output_file)
output_file_button.grid(row=1, column=2)

# Create the preprocess checkbox
preprocess_var = tk.BooleanVar()
preprocess_var.set(True)

preprocess_label = tk.Label(window, text="Preprocess:")
preprocess_label.grid(row=2, column=0)

preprocess_checkbox = tk.Checkbutton(window, variable=preprocess_var, onvalue=True, offvalue=False)
preprocess_checkbox.grid(row=2, column=1)

# Create the threshold input
threshold_var = tk.StringVar()
threshold_var.set(str(DEFAULT_THRESHOLD))

threshold_label = tk.Label(window, text="Threshold (%):")
threshold_label.grid(row=3, column=0)

threshold_entry = tk.Entry(window, textvariable=threshold_var)
threshold_entry.grid(row=3, column=1)

# Create the check button
check_button = tk.Button(window, text="Check", command=batch_check_wrapper)
check_button.grid(row=4, column=1)

# Create the results display
results_label = tk.Label(window, text="Results:")
results_label.grid(row=5, column=0)

results_text = tk.Text(window, height=20, width=80)
results_text.grid(row=6, column=0, columnspan=3)

window.mainloop()
if name == 'main':
# Example usage:
        similarities = batch_check("texts")
for file1, file2, similarity_ratio, matching_text in similarities:
            print(f"{file1} is {similarity_ratio:.2f}% similar to {file2}\nMatching Text:\n{matching_text}\n")
# Run the GUI
run_gui()

#With these additions, the plagiarism checker now has the ability to save batch check results to a CSV file, making it more versatile and useful for a wider range of applications.
