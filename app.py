from flask import Flask, render_template
from jobfit import run_jobfit, ask_llm

app = Flask(__name__)

@app.route("/")
def home(): # Define the home page route
    results, job_texts, indices, chunk_job_rows = run_jobfit()


    best_job = None
    for r in results:
        if r["decision"] == "Apply":
            best_job = r
            break

    company = best_job["company_name"]
    if company == "":
        company = "Company info not provided in job ad"

    retrieved_chunks = []
    top_k = 5
    best_key = best_job["job_key"]

    for idx in indices[0]:
        if chunk_job_rows[idx]["job_key"] == best_key:
            retrieved_chunks.append(job_texts[idx])
        if len(retrieved_chunks) >= top_k:
            break

    context = ""
    context += "Job title: " + str(best_job["job_title"]) + "\n"
    context += "Company: " + str(company) + "\n"
    context += "Relevance score: " + str(best_job["profile_to_job_semantic_relevance"]) + "\n\n"
    context += "Retrieved job chunks:\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += "[Chunk " + str(i) + "]\n"
        context += chunk + "\n\n"

    try:
        why = ask_llm(context)
    except Exception as e:
        why = str(e)

    llm_answer = ""
    llm_answer += "Best matching job: " + str(best_job["job_title"]) + "\n"
    llm_answer += "Company: " + str(company) + "\n"
    llm_answer += "Relevance score: " + str(best_job["profile_to_job_semantic_relevance"]) + "\n\n"
    llm_answer += why

    return render_template("index.html", results=results, llm_answer=llm_answer)

if __name__ == "__main__":
    app.run(debug=True)
