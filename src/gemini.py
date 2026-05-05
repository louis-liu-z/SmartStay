import json
import pandas as pd


def gemini_explain(listings_df: pd.DataFrame, user_prefs: dict, api_key: str) -> list[str]:
    n = len(listings_df)
    fallback = lambda msg: [msg] * n

    try:
        import google.generativeai as genai
    except ImportError:
        return fallback("google-generativeai is not installed. Run: pip install google-generativeai")

    genai.configure(api_key=api_key)
    try:
        model = next(
            (genai.GenerativeModel(m.name)
             for m in genai.list_models()
             if "generateContent" in m.supported_generation_methods),
            None,
        )
    except Exception as e:
        return fallback(f"Could not connect to Gemini: {e}")

    if model is None:
        return fallback("No Gemini model found that supports generateContent.")

    details = []
    for i, (_, row) in enumerate(listings_df.iterrows()):
        amenities_preview = str(row.get("amenities_clean", ""))[:200]
        details.append(
            f"Listing {i + 1}:\n"
            f"- Name: {row['name']}\n"
            f"- City: {row['City']}\n"
            f"- Room type: {row['room_type']}\n"
            f"- Price: ${row['price']:.0f}/night\n"
            f"- Rating: {row['review_scores_rating']}/5\n"
            f"- Amenities: {amenities_preview}"
        )

    prompt = f"""
A user is looking for an Airbnb in California with these preferences:
- Budget: ${user_prefs.get('budget')} per night
- Location: {user_prefs.get('city')}
- Room type: {user_prefs.get('room_type')}
- Must-have amenities: {user_prefs.get('amenities')}

Here are {n} recommended listings:
{chr(10).join(details)}

Return ONLY a JSON array of exactly {n} strings. Each string is a concise 2-3 sentence explanation
of why that listing matches the user's needs — mention price, rating, or specific amenities.
Format: ["Explanation for listing 1.", "Explanation for listing 2.", ...]
No extra text outside the JSON array.
"""
    try:
        raw = model.generate_content(prompt).text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        explanations = json.loads(raw.strip())
        if isinstance(explanations, list):
            return [str(e) for e in explanations[:n]]
        return fallback("Unexpected response format from Gemini.")
    except Exception as e:
        return fallback(f"Gemini error: {e}")
