#pip install ollama httpx

import json
import httpx
import ollama

# Replace with your actual deployed Supabase Edge Function URL
SUPABASE_FUNCTION_URL = "https://<your-project-ref>.supabase.co/functions/v1/get-stock-data"

# 1. Define the tool function that fetches data from your Edge Function
def fetch_stock_data_from_supabase(ticker: str, sections: list[str] = None) -> dict:
    """Calls the Supabase Edge Function API to scrape TickerGate."""
    if not sections:
        sections = ["overview", "forecast"]
    
    payload = {
        "ticker": ticker.upper(),
        "sections": sections
    }
    
    try:
        response = httpx.post(SUPABASE_FUNCTION_URL, json=payload, timeout=20.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to retrieve data: {str(e)}"}

# 2. Describe the Tool Schema for Ollama
stock_data_tool = {
    "type": "function",
    "function": {
        "name": "fetch_stock_data",
        "description": "Get fundamental, financial, analyst forecasts, revenue, and ownership data for a stock ticker from TickerGate.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, TSLA, NVDA",
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["overview", "forecast", "revenue", "ownership", "financials", "all"],
                    },
                    "description": "The specific sections of data required to answer the question.",
                },
            },
            "required": ["ticker"],
        },
    },
}

# 3. Main conversation loop with Ollama
def query_ollama_with_tools(user_prompt: str, model_name: str = "llama3.1:latest"):
    print(f"\nUser: {user_prompt}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a senior financial analyst. Answer user questions using the provided stock data retrieval tools."
        },
        {"role": "user", "content": user_prompt}
    ]

    # Initial request to Ollama
    response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=[stock_data_tool]
    )

    # Check if Ollama decided to call the tool
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            func_name = tool_call.function.name
            args = tool_call.function.arguments

            if func_name == "fetch_stock_data":
                ticker = args.get("ticker")
                sections = args.get("sections", ["overview", "forecast"])
                
                print(f"[Tool Execution] Calling Edge Function for ticker '{ticker}' with sections: {sections}")
                api_result = fetch_stock_data_from_supabase(ticker=ticker, sections=sections)

                # Append assistant tool call and tool result to message history
                messages.append(response.message)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(api_result),
                })

        # Final inference: Ollama analyzes the returned data and answers
        final_response = ollama.chat(
            model=model_name,
            messages=messages
        )
        print(f"\nAI Response:\n{final_response.message.content}")
    else:
        print(f"\nAI Response:\n{response.message.content}")

if __name__ == "__main__":
    # Test queries:
    query_ollama_with_tools("What is Apple's current price target and analyst consensus?")
    query_ollama_with_tools("Summarize the revenue breakdown and major institutional ownership for AAPL.")
