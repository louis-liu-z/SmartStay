import pandas as pd


def gemini_explain(listings_df: pd.DataFrame, user_prefs: dict, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        return "google-generativeai is not installed. Run: pip install google-generativeai"

    genai.configure(api_key=api_key)
    try:
        model = next(
            (genai.GenerativeModel(m.name)
             for m in genai.list_models()
             if "generateContent" in m.supported_generation_methods),
            None,
        )
    except Exception as e:
        return f"Could not connect to Gemini: {e}"

    if model is None:
        return "No Gemini model found that supports generateContent."

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

Here are {len(listings_df)} recommended listings:
{chr(10).join(details)}

For each listing, write a concise 2-3 sentence explanation of why it matches the user's needs.
Be specific — mention price, rating, or amenities that align with their preferences.
Present the explanations as a numbered list matching the order above.
"""
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini error: {e}"
