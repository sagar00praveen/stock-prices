from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/jobinterview')
def jobinterview():
    return render_template('job_interview.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    questions = [
        {"question": "What is the capital of France?", "options": ["A. Berlin", "B. Paris", "C. Madrid", "D. Rome"], "answer": "B"},
        {"question": "What is 2+2?", "options": ["A. 4", "B. 5", "C. 6", "D. 7"], "answer": "A"},
        {"question": "What is the chemical symbol for water?", "options": ["A. H2O", "B. CO2", "C. NaCl", "D. O2"], "answer": "A"},
        {"question": "Who wrote 'To Kill a Mockingbird'?", "options": ["A. Harper Lee", "B. J.K. Rowling", "C. Ernest Hemingway", "D. Mark Twain"], "answer": "A"},
        {"question": "What is the largest planet in our solar system?", "options": ["A. Earth", "B. Mars", "C. Jupiter", "D. Saturn"], "answer": "C"},
        {"question": "Which element has the atomic number 1?", "options": ["A. Hydrogen", "B. Helium", "C. Oxygen", "D. Carbon"], "answer": "A"},
        {"question": "What is the speed of light?", "options": ["A. 300,000 km/s", "B. 150,000 km/s", "C. 200,000 km/s", "D. 250,000 km/s"], "answer": "A"},
        {"question": "Who painted the Mona Lisa?", "options": ["A. Vincent van Gogh", "B. Pablo Picasso", "C. Leonardo da Vinci", "D. Claude Monet"], "answer": "C"},
        {"question": "What is the smallest unit of life?", "options": ["A. Atom", "B. Molecule", "C. Cell", "D. Organ"], "answer": "C"},
        {"question": "Who discovered penicillin?", "options": ["A. Alexander Fleming", "B. Marie Curie", "C. Isaac Newton", "D. Albert Einstein"], "answer": "A"}
    ]
    
    if request.method == 'POST':
        answers = []
        for i in range(len(questions)):
            answer = request.form.get(f'question{i+1}')
            if answer is None:
                answers.append("")
            else:
                answers.append(answer)
        
        correct_answers = [q['answer'] for q in questions]
        results = [answers[i] == correct_answers[i] for i in range(len(correct_answers))]
        return render_template('test_results.html', results=results, questions=questions, answers=answers, correct_answers=correct_answers)
    
    return render_template('test.html', questions=questions)

@app.route('/skilldevelopment')
def skilldevelopment():
    return render_template('skill_development.html')

@app.route('/ai_ml')
def ai_ml():
    return render_template('ai_ml.html')

@app.route('/cloud_computing')
def cloud_computing():
    return render_template('cloud_computing.html')

@app.route('/data_science')
def data_science():
    return render_template('data_science.html')

@app.route('/cybersecurity')
def cybersecurity():
    return render_template('cybersecurity.html')

@app.route('/soc')
def soc():
    return render_template('soc.html')

if __name__ == '__main__':
    app.run(debug=True)
