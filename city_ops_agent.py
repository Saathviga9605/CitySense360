# =====================================================
# CitySense360 - Agentic AI City Operations Brain
# ReAct-style Agent with Tool Use & Reasoning
# Single-file implementation
# =====================================================

from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import datetime
import re

# -----------------------------------------------------
# 1. LOAD LLM (LOCAL, NO API)
# -----------------------------------------------------
print("Loading Agent LLM...")

model_name = "google/flan-t5-base"  # Using base for faster loading
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------------------------------
# 2. MOCKED CITY TOOLS (REPRESENT YOUR MODELS)
# -----------------------------------------------------

@tool
def traffic_monitor(query: str) -> str:
    """Checks city traffic congestion status"""
    congestion = random.choice(["Low", "Moderate", "High"])
    return f"Traffic congestion level is {congestion}."

@tool
def air_quality_monitor(query: str) -> str:
    """Retrieves current air quality index"""
    aqi = random.randint(60, 180)
    status = "Good" if aqi < 100 else "Moderate" if aqi < 150 else "Poor"
    return f"Current AQI is {aqi} ({status})."

@tool
def energy_monitor(query: str) -> str:
    """Checks smart grid power load conditions"""
    load = random.choice(["Normal", "Peak Load", "Overload Risk"])
    return f"Smart grid load status: {load}."

@tool
def citizen_complaints(query: str) -> str:
    """Fetches recent citizen complaints"""
    complaints = [
        "Water supply disruption in Zone 3",
        "Street light failure reported in Zone 5",
        "Garbage overflow near market area"
    ]
    return f"Recent complaints: {', '.join(complaints)}."

# -----------------------------------------------------
# 3. SIMPLE AGENT IMPLEMENTATION
# -----------------------------------------------------

tools = [
    traffic_monitor,
    air_quality_monitor,
    energy_monitor,
    citizen_complaints
]

class SimpleReActAgent:
    def __init__(self, llm, tools, max_iterations=5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        
    def run(self, query):
        """Execute the agent with ReAct-style reasoning"""
        
        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        print(f"\n🤖 Agent Query: {query}\n")
        
        # Collect all tool results
        results = []
        for tool_name, tool in self.tools.items():
            try:
                print(f"🔧 Executing: {tool_name}")
                result = tool.invoke("")
                results.append(f"{tool_name}: {result}")
                print(f"   ✓ {result}")
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results.append(f"{tool_name}: Error occurred")
        
        # Generate summary with LLM
        summary_prompt = f"""
Based on the following city data, create a brief summary report:

{chr(10).join(results)}

Provide a concise 3-4 sentence summary with any priority actions needed.
"""
        
        try:
            print("\n📝 Generating summary...")
            summary = self.llm.invoke(summary_prompt)
            return summary
        except Exception as e:
            print(f"Summary generation error: {e}")
            return "\n".join(results)

# -----------------------------------------------------
# 4. INITIALIZE AGENT
# -----------------------------------------------------

agent = SimpleReActAgent(
    llm=llm,
    tools=tools,
    max_iterations=5
)

# -----------------------------------------------------
# 5. CITY OPERATIONS REPORT GENERATOR
# -----------------------------------------------------

def generate_city_report():
    today = datetime.date.today().strftime("%d %B %Y")

    prompt = f"""
Generate a city operations status report for {today}.

Check:
- Traffic conditions
- Air quality
- Energy grid status  
- Citizen complaints

Provide summary and priority actions.
"""

    print("\n" + "="*60)
    print(f"CitySense360 - City Operations Report | {today}")
    print("="*60)
    
    report = agent.run(prompt)
    
    return report

# -----------------------------------------------------
# 6. RUN AGENT
# -----------------------------------------------------

if __name__ == "__main__":
    report = generate_city_report()
    
    print("\n" + "-"*60)
    print("📊 FINAL REPORT")
    print("-"*60)
    print(report)
    print("\n" + "="*60 + "\n")