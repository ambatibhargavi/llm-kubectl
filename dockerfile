FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY llm_model.py .
COPY Salary_Data_Based_country_and_race.csv .
CMD ["python", "./llm_model.py"]
