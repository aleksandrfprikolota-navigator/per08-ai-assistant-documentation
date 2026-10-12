"""
Модуль работы с векторным хранилищем ChromaDB для GigaChat.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from gigachat_client import GigaChatClient

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB с GigaChat embeddings."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # GigaChat клиент для создания embeddings.
        # Если embeddings API Сбера недоступен, индекс строится через OpenAI,
        # чтобы размерность запроса и документов совпадала.
        self.gigachat_client = GigaChatClient()
        self.embedding_backend = "gigachat"
        self.openai_client = None
    
    def _prepare_paragraphs(self, text: str, chunk_size: int) -> List[str]:
        """Готовит абзацы: длинные блоки со скачанных страниц режет по строкам."""
        text = text.replace("\r\n", "\n").replace("\xa0", " ")
        paragraphs = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) > chunk_size and "\n" in para:
                paragraphs.extend(line.strip() for line in para.split("\n") if line.strip())
            else:
                paragraphs.append(para)
        return paragraphs

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Длинный абзац со строк сайта режется по строкам, затем по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        paragraphs = self._prepare_paragraphs(text, chunk_size)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                continue

            overlap_text = ""
            if current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = ""

            if len(paragraph) > chunk_size:
                sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                if overlap_text and sentence_chunks:
                    sentence_chunks[0] = (overlap_text + "\n\n" + sentence_chunks[0]).strip()
                if sentence_chunks:
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1]
                continue

            current_chunk = (overlap_text + "\n\n" + paragraph).strip() if overlap_text else paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _resolve_sources(self, file_path: str) -> List[Path]:
        """Файл или каталог с .txt. Относительный путь ищется и от текущей папки, и от модуля."""
        path = Path(file_path)
        if not path.exists():
            alt = Path(__file__).parent / file_path
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Путь {file_path} не найден")
        if path.is_dir():
            files = sorted(path.glob("*.txt"))
            if not files:
                raise FileNotFoundError(f"В каталоге {path} нет txt-файлов")
            return files
        return [path]

    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла или каталога data/ в векторное хранилище.
        
        Args:
            file_path: путь к txt-файлу или к папке с документами компании
        """
        files = self._resolve_sources(file_path)
        
        chunks = []
        sources = []
        for file in files:
            text = file.read_text(encoding="utf-8")
            file_chunks = self._chunk_text(text)
            print(f"{file.name}: {len(file_chunks)} чанков")
            chunks.extend(file_chunks)
            sources.extend([file.name] * len(file_chunks))
        
        print(f"Всего чанков: {len(chunks)} из {len(files)} файлов")
        
        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return
        
        batch_size = 16
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_sources = sources[start:start + batch_size]
            batch_embeddings = self._create_embeddings(batch_chunks)
            self.collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                ids=[f"doc_{start + i}" for i in range(len(batch_chunks))],
                metadatas=[{"source": source} for source in batch_sources]
            )
            print(f"Обработано {min(start + batch_size, len(chunks))}/{len(chunks)} чанков")
        
        print(f"Загружено {len(chunks)} документов в коллекцию '{self.collection_name}'")
    
    def _use_openai_embeddings(self):
        """Переключает векторизацию на OpenAI, если embeddings GigaChat недоступны."""
        if self.embedding_backend == "openai":
            return
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "Embeddings GigaChat недоступны, а OPENAI_API_KEY не задан"
            )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_backend = "openai"
        print("Embeddings GigaChat недоступны. Индекс строится через OpenAI text-embedding-3-small.")

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Пакетная векторизация. Один и тот же backend используется для всех чанков и запросов."""
        if self.embedding_backend == "openai":
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]

        try:
            return self.gigachat_client.get_embeddings(texts)
        except Exception as error:
            print(f"Ошибка embeddings GigaChat: {error}")
            self._use_openai_embeddings()
            return self._create_embeddings(texts)

    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        return self._create_embeddings([text])[0]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            metadatas = results.get('metadatas') or [[]]
            for i in range(len(results['documents'][0])):
                meta = metadatas[0][i] if metadatas and metadatas[0] else {}
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'source': (meta or {}).get('source', '')
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("GIGACHAT_AUTH_KEY") or not os.getenv("GIGACHAT_RQUID"):
        print("Ошибка: установите GIGACHAT_AUTH_KEY и GIGACHAT_RQUID в .env")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="gigachat_rag_collection")
    
    vector_store.load_documents("data")
    
    # Поиск по базе компании «1С-Премиум»
    results = vector_store.search("Какой график работы офиса 1С-Премиум?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

