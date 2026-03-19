# Upload all the files present in Test_Docs before running the codes
## https://console.groq.com/keys
> Create an api from the above link and add to line 40 of the 2 code files at the placeholder:
```LLM_CLIENT = OpenAI(
    api_key  = os.environ.get("GROQ_API_KEY", "<groq_api>"),
    base_url = "https://api.groq.com/openai/v1",
)
```
Refer to [Documentation](documentation.md) for more details
