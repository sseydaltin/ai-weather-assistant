import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool


load_dotenv()


def _format_weather_output(data: dict) -> str:
    name = data.get("name") or data.get("sys", {}).get("country", "")
    weather = (data.get("weather") or [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})

    emoji = {
        "Thunderstorm": "⛈️",
        "Drizzle": "🌦️",
        "Rain": "🌧️",
        "Snow": "❄️",
        "Clear": "☀️",
        "Clouds": "☁️",
    }.get(weather.get("main", ""), "🌍")

    lines = [
        f"{emoji} {name} Hava Durumu",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Açıklama: {weather.get('description', '').capitalize()}",
        f"Sıcaklık: {main.get('temp', 'N/A')}°C (Hissedilen: {main.get('feels_like', 'N/A')}°C)",
        f"Nem: {main.get('humidity', 'N/A')}%",
        f"Rüzgar: {wind.get('speed', 'N/A')} m/s",
        f"Basınç: {main.get('pressure', 'N/A')} hPa",
    ]

    if "visibility" in data:
        lines.append(f"Görüş: {int(data['visibility'])/1000:.1f} km")

    sys = data.get("sys", {})
    if sys.get("sunrise") and sys.get("sunset"):
        try:
            sunrise = time.strftime("%H:%M", time.localtime(sys["sunrise"]))
            sunset = time.strftime("%H:%M", time.localtime(sys["sunset"]))
            lines.append(f"Gündoğumu/Günbatımı: {sunrise} / {sunset}")
        except Exception:
            pass

    return "\n".join(lines)


def _call_openweather(city: str, api_key: str, timeout: Optional[float]) -> dict:
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "tr"}
    resp = requests.get(base_url, params=params, timeout=timeout)

    if resp.status_code == 401:
        raise PermissionError("OpenWeather API key geçersiz (401). .env içindeki OPENWEATHER_API_KEY'i kontrol et.")
    if resp.status_code == 404:
        raise ValueError(f"Şehir bulunamadı (404): {city}")
    if resp.status_code == 429:
        raise RuntimeError("OpenWeather rate limit aşıldı (429). Lütfen daha sonra tekrar deneyin.")

    resp.raise_for_status()
    return resp.json()


@tool
def get_current_weather(city: str) -> str:
    """
    Belirtilen şehir için güncel hava durumunu getirir (OpenWeatherMap).

    Args:
        city: Şehir adı (ör. "Istanbul", "London")

    Returns:
        Emoji'li ve okunabilir formatta bir string.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "❌ OPENWEATHER_API_KEY tanımlı değil. .env dosyanızı doldurun."

    timeout_s = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))

    try:
        data = _call_openweather(city=city, api_key=api_key, timeout=timeout_s)
        return _format_weather_output(data)
    except requests.Timeout:
        return "⏱️ OpenWeather isteği zaman aşımına uğradı. Biraz sonra tekrar deneyin."
    except PermissionError as e:
        return f"❌ {e}"
    except ValueError as e:
        return f"❌ {e}"
    except RuntimeError as e:
        return f"⚠️ {e}"
    except requests.RequestException as e:
        return f"❌ OpenWeather isteği başarısız: {e}"


if __name__ == "__main__":
    print("=" * 60)
    print("WEATHER TOOL TEST")
    print("=" * 60)

    for c in ["Istanbul", "London", "XyzNotARealCity"]:
        print(f"\n❓ {c} için hava durumu")
        print("-" * 60)
        print(get_current_weather.invoke(c))


