from flask import Flask, render_template, request, redirect
import csv
import os

app = Flask(__name__)

# Define the CSV file path
CSV_FILE_PATH = 'feedback_data.csv'

# Function to write feedback to CSV
def write_to_csv(category, comment, rating):
    # Check if file exists, if not, create it and write the header
    file_exists = os.path.isfile(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Category', 'Comment', 'Rating'])  # Write the header
        writer.writerow([category, comment, rating])

@app.route('/', methods=['GET', 'POST'])
def feedback_form():
    if request.method == 'POST':
        # Get the form data
        category = request.form['category']
        comment = request.form['comment']
        rating = request.form['rating']

        # Write to CSV file
        write_to_csv(category, comment, rating)

        # Redirect to a thank you page or back to the form
        return redirect('/thank-you')

    return render_template('index.html')  # Render the HTML form
@app.route('/thank-you')
def thank_you():
    return "Thank you for your feedback! It has been recorded."

if __name__ == '__main__':
    app.run(debug=True, port=5001)
