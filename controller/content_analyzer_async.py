"""
content_analyzer_async.py

Asynchronous content analysis utilities used by both the API and Gradio app.

- Chunking long PDF text into GPT-sized chunks
- Analyzing content (key points + overall summary) across chunks
- Summarizing the generated podcast transcript for context
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from controller.config import (
	get_openai_api_key,
	get_openai_model,
	CHUNK_SIZE,
)


# Limit concurrent requests to avoid rate limits
MAX_CONCURRENT_REQUESTS = 3
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def chunk_text_for_gpt(text: str, max_tokens: int = 4000) -> List[str]:
	"""
	Naive text chunker using character counts as an approximation for tokens.
	Uses CHUNK_SIZE from config as the base size and respects max_tokens as an upper bound.
	"""
	if not text:
		return []

	# Simple heuristic: ~4 chars per token → keep chunks conservatively small
	approx_chars_per_token = 4
	max_chars = min(CHUNK_SIZE, max_tokens) * approx_chars_per_token
	max_chars = max(2000, max_chars)  # ensure a reasonable minimum

	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(start + max_chars, len(text))
		# Try to break at a paragraph boundary if possible
		slice_text = text[start:end]
		last_break = slice_text.rfind("\n\n")
		if last_break > 500:  # keep chunks reasonably sized, avoid tiny last piece
			end = start + last_break
			slice_text = text[start:end]
		chunks.append(slice_text.strip())
		start = end

	# Filter any empties
	return [c for c in chunks if c]


@retry(
	stop=stop_after_attempt(3),
	wait=wait_exponential(multiplier=1, min=2, max=10),
	retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def _openai_json_call(
	session: aiohttp.ClientSession,
	prompt: str,
	response_schema: Dict[str, Any],
	max_tokens: int = 512,
	temperature: float = 0.3,
) -> Dict[str, Any]:
	"""
	Make an async call to OpenAI Chat Completions API expecting JSON via response_format json_schema.
	"""
	async with API_SEMAPHORE:
		headers = {
			"Authorization": f"Bearer {get_openai_api_key()}",
			"Content-Type": "application/json",
		}
		payload = {
			"model": get_openai_model(),
			"messages": [{"role": "user", "content": prompt}],
			"max_tokens": max_tokens,
			"temperature": temperature,
			"response_format": {"type": "json_schema", "json_schema": response_schema},
		}
		async with session.post(
			"https://api.openai.com/v1/chat/completions",
			headers=headers,
			json=payload,
			timeout=45,
		) as resp:
			if resp.status == 200:
				result = await resp.json()
				content = result["choices"][0]["message"]["content"]
				return json.loads(content)
			else:
				raise aiohttp.ClientError(f"OpenAI error {resp.status}: {await resp.text()}")


async def _analyze_chunk(session: aiohttp.ClientSession, chunk: str) -> Dict[str, Any]:
	"""
	Analyze a single text chunk to extract key points and a micro-summary.
	"""
	prompt = (
		"You will analyze a chunk of a longer document. "
		"Extract 5-10 concise key points and a 2-3 sentence micro_summary capturing the essence.\n\n"
		f"Chunk:\n{chunk}\n\n"
	)
	schema = {
		"name": "chunk_analysis",
		"strict": True,
		"schema": {
			"type": "object",
			"properties": {
				"key_points": {"type": "array", "items": {"type": "string"}},
				"micro_summary": {"type": "string"},
			},
			"required": ["key_points", "micro_summary"],
			"additionalProperties": False,
		},
	}
	result = await _openai_json_call(session, prompt, schema, max_tokens=500, temperature=0.2)
	return {
		"key_points": result.get("key_points", []),
		"micro_summary": result.get("micro_summary", ""),
	}


async def analyze_content_completely_async(text_chunks: List[str]) -> Dict[str, Any]:
	"""
	Analyze all text chunks concurrently and produce an overall summary and merged key points.
	Returns a dict with keys: key_points (List[str]), summary (str).
	"""
	if not text_chunks:
		return {"key_points": [], "summary": ""}

	async with aiohttp.ClientSession() as session:
		chunk_tasks = [
			_analyze_chunk(session, chunk)
			for chunk in text_chunks
		]
		chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

		all_points: List[str] = []
		micro_summaries: List[str] = []
		for res in chunk_results:
			if isinstance(res, Exception):
				continue
			all_points.extend([p for p in res.get("key_points", []) if isinstance(p, str) and p.strip()])
			if res.get("micro_summary"):
				micro_summaries.append(res["micro_summary"]) 

		# Produce an overall summary from the micro summaries
		overall_prompt = (
			"You will receive multiple micro summaries from different parts of a single document. "
			"Write a cohesive, 1-2 paragraph overall summary that captures the document's purpose, main arguments, and conclusions.\n\n"
			f"Micro summaries:\n- " + "\n- ".join(micro_summaries[:30])
		)
		overall_schema = {
			"name": "overall_summary",
			"strict": True,
			"schema": {
				"type": "object",
				"properties": {"summary": {"type": "string"}},
				"required": ["summary"],
				"additionalProperties": False,
			},
		}
		try:
			final = await _openai_json_call(session, overall_prompt, overall_schema, max_tokens=350, temperature=0.3)
			summary_text = final.get("summary", "")
		except Exception:
			summary_text = "\n".join(micro_summaries[:5])

		# De-duplicate key points while preserving order
		seen = set()
		unique_points: List[str] = []
		for p in all_points:
			if p not in seen:
				seen.add(p)
				unique_points.append(p)

		return {"key_points": unique_points[:30], "summary": summary_text.strip()}


async def summarize_transcript_async(script: List[Dict[str, str]]) -> str:
	"""
	Summarize a full podcast transcript (list of {speaker, text}) as a concise overview:
	- How the podcast starts
	- Topic flow/segments in order
	- Key takeaways and any disagreements
	- How it concludes
	Returns a markdown-formatted string suitable for UI/API.
	"""
	if not script:
		return ""

	# Trim very long transcripts by sampling turns while keeping order
	max_turns = 300
	selected = script[:max_turns]
	transcript_text = "\n".join([f"{s.get('speaker', 'Host')}: {s.get('text', '').strip()}" for s in selected])

	prompt = (
		"Summarize the following podcast transcript for listeners who want context before listening. "
		"Write 120-200 words. Include: how it opens, the progression of topics, notable points of agreement/disagreement, and how it wraps up. "
		"Use neutral, engaging language and avoid spoilers for exact phrasing. Provide bullet points for topic flow.\n\n"
		f"Transcript:\n{transcript_text}"
	)

	schema = {
		"name": "podcast_summary",
		"strict": True,
		"schema": {
			"type": "object",
			"properties": {
				"summary": {"type": "string"}
			},
			"required": ["summary"],
			"additionalProperties": False
		}
	}

	async with aiohttp.ClientSession() as session:
		try:
			result = await _openai_json_call(session, prompt, schema, max_tokens=300, temperature=0.4)
			return result.get("summary", "").strip()
		except Exception:
			# Fallback: simple heuristic summary if API fails
			opening = selected[0].get("text", "").strip() if selected else ""
			closing = selected[-1].get("text", "").strip() if selected else ""
			return (
				"Podcast overview: a conversation that opens with "
				+ (opening[:120] + ("…" if len(opening) > 120 else ""))
				+ " … and concludes with "
				+ (closing[:120] + ("…" if len(closing) > 120 else ""))
			)