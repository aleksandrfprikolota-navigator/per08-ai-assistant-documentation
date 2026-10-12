"""
Клиент для работы с GigaChat API от Сбера.
Управляет авторизацией и запросами к API.
"""

import requests
import os
import urllib3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class GigaChatClient:
    """Клиент для работы с GigaChat API."""
    
    def __init__(self, auth_key: str = None, rq_uid: str = None):
        """
        Инициализация GigaChat клиента.
        
        Args:
            auth_key: ключ авторизации (Basic token)
            rq_uid: уникальный идентификатор запроса
        """
        self.auth_key = auth_key or os.getenv("GIGACHAT_AUTH_KEY")
        self.rq_uid = rq_uid or os.getenv("GIGACHAT_RQUID")
        
        if not self.auth_key:
            raise ValueError("GIGACHAT_AUTH_KEY не установлен")
        if not self.rq_uid:
            raise ValueError("GIGACHAT_RQUID не установлен")
        
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"
        
        self.access_token = None
        self.token_expires_at = None
        
        # Получаем токен при инициализации
        self._refresh_token()
    
    def _refresh_token(self):
        """Получение нового access token."""
        payload = {'scope': 'GIGACHAT_API_PERS'}
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': self.rq_uid,
            'Authorization': f'Basic {self.auth_key}'
        }
        
        try:
            response = requests.post(
                self.oauth_url,
                headers=headers,
                data=payload,
                verify=False  # Для работы с сертификатом Сбера
            )
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            # Токен действует 30 минут, обновляем за 1 минуту до истечения
            self.token_expires_at = datetime.now() + timedelta(minutes=29)
            
            print("✓ GigaChat access token получен")
            
        except Exception as e:
            raise Exception(f"Ошибка получения access token: {e}")
    
    def _ensure_token_valid(self):
        """Проверка валидности токена и обновление при необходимости."""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            print("⟳ Обновление access token...")
            self._refresh_token()
    
    def _get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запросов."""
        self._ensure_token_valid()
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       model: str = "GigaChat-2-Max",
                       temperature: float = 0.3,
                       max_tokens: int = 500) -> str:
        """
        Отправка запроса к чат-модели GigaChat.
        
        Args:
            messages: список сообщений в формате [{"role": "user", "content": "..."}]
            model: название модели
            temperature: температура генерации
            max_tokens: максимальное количество токенов в ответе
            
        Returns:
            текст ответа от модели
        """
        url = f"{self.api_url}/chat/completions"
        models_to_try = [model]
        for fallback in ("GigaChat-2-Pro", "GigaChat-2-Max"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)
        
        last_error = None
        for current_model in models_to_try:
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    verify=False,
                    timeout=60
                )
            except Exception as error:
                last_error = error
                print(f"{current_model}: ошибка сети, пробую следующую модель...")
                continue

            if response.ok:
                if current_model != model:
                    print(f"Модель {model} недоступна, ответ получен от {current_model}")
                data = response.json()
                return data['choices'][0]['message']['content']

            detail = response.text[:300]
            last_error = Exception(f"{response.status_code} {response.reason}: {detail}")
            if response.status_code >= 500:
                print(f"{current_model} вернул {response.status_code}, пробую следующую модель...")
                continue
            break

        raise Exception(f"Ошибка запроса к GigaChat: {last_error}")
    
    def get_embeddings(self, texts: List[str], model: str = "Embeddings") -> List[List[float]]:
        """
        Получение векторных представлений текстов.
        
        Args:
            texts: список текстов для векторизации
            model: модель для embeddings
            
        Returns:
            список векторов
        """
        url = f"{self.api_url}/embeddings"
        
        payload = {
            "model": model,
            "input": texts
        }
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return [item['embedding'] for item in data['data']]
            
        except Exception as e:
            raise Exception(f"Ошибка embeddings GigaChat: {e}") from e
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных моделей.
        
        Returns:
            список моделей
        """
        url = f"{self.api_url}/models"
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', [])
            
        except Exception as e:
            print(f"Ошибка получения списка моделей: {e}")
            return []


if __name__ == "__main__":
    # Тестирование клиента
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Загружаем .env
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    try:
        client = GigaChatClient()
        
        # Тест получения моделей
        print("\n=== Доступные модели ===")
        models = client.get_models()
        for model in models:
            print(f"- {model.get('id', 'unknown')}")
        
        # Тест чата
        print("\n=== Тест чата ===")
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Ответь одним словом: работает"}
            ]
        )
        print(f"Ответ: {response}")
        
        # Тест embeddings
        print("\n=== Тест embeddings ===")
        embeddings = client.get_embeddings(["Тестовый текст"])
        print(f"Размерность вектора: {len(embeddings[0])}")
        
        print("\n✓ Все тесты пройдены")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

