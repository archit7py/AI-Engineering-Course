import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key not Found")

client = Groq(api_key = my_api_key)
model = "llama-3.3-70b-versatile"

Job_D = """
Project Role : AI / ML Engineer
Project Role Description : Develops applications and systems that utilize AI tools, Cloud AI services, with proper cloud or on-prem application pipeline with production ready quality. Be able to apply GenAI models as part of the solution. Could also include but not limited to deep learning, neural networks, chatbots, image processing.
Must have skills : Machine Learning Operations
Good to have skills : NA
Minimum 5 year(s) of experience is required
Educational Qualification : 15 years full time education

Summary:
As an AI / ML Engineer, you will develop applications and systems that leverage artificial intelligence tools and cloud AI services. Your typical day will involve designing and implementing production-ready application pipelines, ensuring high-quality standards are met. You will also explore the integration of generative AI models into solutions, while working on various aspects of deep learning, neural networks, chatbots, and image processing to enhance functionality and performance.

Roles & Responsibilities:
- Continuously evaluate and improve existing processes to enhance efficiency.
- Engage with multiple teams and contribute on key decisions.
- Provide solutions to problems for their immediate team and across multiple teams.
- Facilitate knowledge sharing sessions to enhance team skills and capabilities.
- Monitor project progress and ensure alignment with strategic goals.

Professional & Technical Skills:
- Strong Engineering experience with advance python skills.
- Design, build, and maintain LLM pipelines for training, fine-tuning, evaluation, and deployment.
- Work on operationalize LLMs, from experimentation to deployment in production environments.
- Implement and monitor observability tools for LLM applications (latency, throughput, prompt performance, hallucinations, drift, etc.).
- Manage prompt management and versioning systems, as well as fine-tuning and retrieval-augmented generation (RAG) workflows.
- Automate model validation, testing, and CI/CD pipelines to support safe and efficient LLM deployment.
- Ensure security, compliance, and ethical use of LLMs in production environments (e.g., data governance, bias detection).
- Stay updated with advances in LLM infrastructure, serving frameworks, and tooling ecosystems.
- Must To Have Skills: Proficiency in Machine Learning Operations.
- Good exposure of cloud based services including AI services.
- Must have thorough understanding of infrastructure need for LLMOps implementation.
- Must have python skills.
- Should have Multi Cloud skills
- Experience with Machine learning frameworks
- Ability to implement and optimize machine learning models for production environments.

Additional Information:
- The candidate should have minimum 5 years of experience in Machine Learning Operations.
- This position is based at our Bengaluru office.
- A 15 years full time education is required.

"""
Resume = """
I am an aspiring AI Engineer currently pursuing a Bachelor of Technology in Computer Science with a specialization in Artificial Intelligence and Machine Learning at Uttarakhand Technical University (2023–2027). Throughout my academic journey, I have secured 2nd rank in both the 4th and 5th semesters, reflecting my strong academic performance. I have gained practical industry experience through a 4-week AI & Data Analytics Virtual Internship with AICTE–Shell India–Edunet Foundation, where I worked with Python, Scikit-learn, TensorFlow, Keras, PyTorch, Pandas, NumPy, and Streamlit to build machine learning workflows. Additionally, I completed an Artificial Intelligence training program at SLOG Solutions Pvt. Ltd., where I developed practical knowledge of AI, machine learning concepts, and data preprocessing techniques. My technical expertise includes Python, C++, Java, Machine Learning, Deep Learning, Computer Vision, Generative AI, TensorFlow, PyTorch, Keras, Scikit-learn, Pandas, NumPy, Flask, PydanticAI, Streamlit, Git, GitHub, VS Code, and Jupyter Notebook, along with a solid understanding of Data Structures & Algorithms, Object-Oriented Programming, Database Management Systems, and Operating Systems. I have built several AI-focused projects, including an AI-powered Resume Evaluator using PydanticAI, Groq API, and FastAPI that analyzes resumes against job descriptions and generates ATS-style feedback; an EV Adoption Forecast Tool that predicts EV charging station growth using machine learning, feature engineering, and Scikit-learn; and a CIFAR-10 Image Classifier built with PyTorch and Convolutional Neural Networks, achieving 82% test accuracy. I am passionate about developing scalable AI-powered applications and continuously expanding my expertise in machine learning, deep learning, and generative AI to solve real-world problems.
"""
def ask_llm(system_prompt, user_prompt):
    sys_message = {
        "role" : "system",
        "content" : system_prompt
    }
    user_msg = {
        "role" : "user",
        "content" : user_prompt
    }
    messages = [sys_message,user_msg]
    response  = client.chat.completions.create(model = model , messages = messages)
    answer  = response.choices[0].message.content
    return answer

def step_1_res_extract(Resume):
    print("STEP 1")
    # extract skils from resume 
    system_prompt = """
    You are a professional HR Assistant , Extract the skils from the Resume Provided, Only return the skills no other information, Do not invent any skills by Yourself
"""
    user_prompt = f"""
    Extract the skills from the resume
    {Resume}
"""
    return ask_llm(system_prompt, user_prompt)

def step_2_jd_extract(Job_D):
    print("STEP 2")
    # extract skils from resume 
    system_prompt = """
    You are a professional HR Assistant , Extract the skils from the Job Description Provided, Only return the skills no other information, Do not invent any skills by Yourself
    """
    user_prompt = f"""
    Extract the skills from the resume
    {Job_D}
    """
    return ask_llm(system_prompt, user_prompt)

def step3_match(candidate,jd):
    print("STEP 3")
    system_prompt = """
    You are a professional HR Assistant. Compare the skills of the resume given and the skills required in the Job_D and produce a final score between 1 and 100. Also produce a short verdict whether the candiadate is a good fit for the role
    """
    user_prompt = f"""
    Compare and match the skills
    Job_D:
    {jd}
    Candidate:
    {candidate}
    """
    return ask_llm(system_prompt,user_prompt)

candidate = step_1_res_extract(Resume)
print(candidate)
sleep(2)
jd = step_2_jd_extract(Job_D)
print(jd)
sleep(2)
score = step3_match(candidate,jd)
print(score)
