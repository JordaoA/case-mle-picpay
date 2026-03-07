"""
extractor.py
------------
Responsible for fetching and normalizing data from PokeAPI.
Designed to be environment-agnostic: works locally and on Databricks.

Usage:
    from extractor import PokeAPIExtractor
    extractor = PokeAPIExtractor()
    raw_data = extractor.fetch_all_pokemon()
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pokeapi.extractor")


@dataclass
class PokemonRecord:
    pokemon_id: int
    name: str
    height: int
    weight: int
    base_experience: Optional[int]


@dataclass
class PokemonType:
    pokemon_id: int
    type_name: str


@dataclass
class PokemonStat:
    pokemon_id: int
    stat_name: str
    base_stat: int


@dataclass
class PokemonAbility:
    pokemon_id: int
    ability_name: str
    is_hidden: bool


@dataclass
class ExtractedData:
    pokemon: list[PokemonRecord] = field(default_factory=list)
    types: list[PokemonType] = field(default_factory=list)
    stats: list[PokemonStat] = field(default_factory=list)
    abilities: list[PokemonAbility] = field(default_factory=list)


class PokeAPIExtractor:
    BASE_URL = "https://pokeapi.co/api/v2"
    DEFAULT_PAGE_SIZE = 100
    DEFAULT_MAX_WORKERS = 20
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_BACKOFF = 2.0

    def __init__(
        self,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        self.page_size = page_size
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.session = self._build_session()

    def fetch_all_pokemon(self, limit: Optional[int] = None) -> ExtractedData:
        """
        Main entry point.
        Fetches the full pokemon list and then each pokemon's detail page
        concurrently.

        Args:
            limit: Optional cap on the number of pokémons (useful for testing).

        Returns:
            ExtractedData with all four normalized tables populated.
        """
        urls = self._fetch_pokemon_url_list(limit=limit)
        logger.info(f"Starting detail fetch for {len(urls)} pokémons "
                    f"with {self.max_workers} workers...")

        result = ExtractedData()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_pokemon_detail, url): url
                for url in urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    data = future.result()
                    if data:
                        result.pokemon.append(data["pokemon"])
                        result.types.extend(data["types"])
                        result.stats.extend(data["stats"])
                        result.abilities.extend(data["abilities"])
                except Exception as exc:
                    logger.warning(f"Failed to process {url}: {exc}")
                finally:
                    completed += 1
                    if completed % 100 == 0:
                        logger.info(f"Progress: {completed}/{len(urls)} pokémons processed")

        logger.info(
            f"Extraction complete — "
            f"{len(result.pokemon)} pokémons | "
            f"{len(result.types)} types | "
            f"{len(result.stats)} stats | "
            f"{len(result.abilities)} abilities"
        )
        return result

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        return session

    def _get_with_retry(self, url: str) -> dict:
        """GET with exponential backoff retry logic."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == self.retry_attempts:
                    raise
                wait = self.retry_backoff * (2 ** (attempt - 1))
                logger.debug(
                    f"Attempt {attempt} failed for {url}: {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

    def _fetch_pokemon_url_list(self, limit: Optional[int] = None) -> list[str]:
        """
        Paginates through /pokemon and collects all detail URLs.
        PokeAPI returns a 'next' cursor — we follow it until None.
        """
        urls = []
        endpoint = f"{self.BASE_URL}/pokemon?limit={self.page_size}&offset=0"

        while endpoint:
            logger.info(f"Fetching page: {endpoint}")
            data = self._get_with_retry(endpoint)
            for item in data.get("results", []):
                urls.append(item["url"])
            endpoint = data.get("next")  # None when last page

        if limit:
            urls = urls[:limit]

        logger.info(f"Found {len(urls)} pokémon URLs total")
        return urls

    def _extract_id_from_url(self, url: str) -> int:
        """Extracts the numeric ID from a PokeAPI URL."""
        # URL format: https://pokeapi.co/api/v2/pokemon/1/
        return int(url.rstrip("/").split("/")[-1])

    def _fetch_pokemon_detail(self, url: str) -> Optional[dict]:
        """
        Fetches a single pokemon detail page and normalizes it
        into our four table structures.
        """
        raw = self._get_with_retry(url)
        pokemon_id = self._extract_id_from_url(url)

        pokemon = PokemonRecord(
            pokemon_id=pokemon_id,
            name=raw["name"],
            height=raw["height"],
            weight=raw["weight"],
            base_experience=raw.get("base_experience"),  # can be null
        )

        types = [
            PokemonType(
                pokemon_id=pokemon_id,
                type_name=t["type"]["name"],
            )
            for t in raw.get("types", [])
        ]

        stats = [
            PokemonStat(
                pokemon_id=pokemon_id,
                stat_name=s["stat"]["name"],
                base_stat=s["base_stat"],
            )
            for s in raw.get("stats", [])
        ]

        abilities = [
            PokemonAbility(
                pokemon_id=pokemon_id,
                ability_name=a["ability"]["name"],
                is_hidden=a["is_hidden"],
            )
            for a in raw.get("abilities", [])
        ]

        return {
            "pokemon": pokemon,
            "types": types,
            "stats": stats,
            "abilities": abilities,
        }