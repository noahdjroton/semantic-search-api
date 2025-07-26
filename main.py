from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
import numpy as np
from typing import Optional, List, Dict
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Search API", version="2.0.0", description="Enhanced semantic search API")

# CORS configuration to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for thresholds
SIMILARITY_THRESHOLD = 0.5
MAX_RESULTS = 10
MIN_SIMILARITY_FOR_DISPLAY = 0.15

# Load AI model
logger.info("Loading sentence transformers model...")
# all-MiniLM-L6-v2 / 
model = SentenceTransformer('all-mpnet-base-v2')
logger.info("Model loaded successfully!")

# Sample products dataset
products = [
    # Women's clothing
    "Robe d'été légère en coton blanc",
    "Robe de soirée élégante noire",
    "Robe midi fleurie printemps",
    "Pull chaud en laine mérinos rouge",
    "Pull col roulé cachemire beige",
    "Cardigan long en tricot gris",
    "Jupe plissée courte noire",
    "Jupe longue bohème colorée",
    "Blouse en soie crème",
    "Chemisier rayé bleu blanc",
    
    # Men's clothing
    "Chemise blanche classique coton",
    "Polo sport respirant marine",
    "T-shirt basic en coton bio noir",
    "T-shirt graphique vintage",
    "Sweat à capuche confortable gris",
    "Veste en cuir vintage marron",
    "Blazer costume élégant bleu",
    "Pantalon chino beige",
    "Jeans slim stretch confortable",
    "Short cargo kaki",
    
    # Shoes
    "Baskets running respirantes blanches",
    "Sneakers lifestyle urbaines",
    "Chaussures de ville cuir noir",
    "Bottes d'hiver imperméables",
    "Bottines chelsea cuir marron",
    "Sandales d'été élégantes",
    "Tongs plage colorées",
    "Escarpins talons hauts noirs",
    "Mocassins confort cuir",
    "Chaussures sport fitness",
    
    # Accessories
    "Écharpe en cachemire douce",
    "Bonnet laine tricot",
    "Casquette baseball ajustable",
    "Ceinture cuir véritable",
    "Sac à main cuir élégant",
    "Sac à dos voyage résistant",
    "Portefeuille cuir compact",
    "Montre sport étanche",
    "Lunettes de soleil polarisées",
    "Bijoux fantaisie dorés",
    
    # Sportswear
    "Short de bain coloré",
    "Maillot de bain une pièce",
    "Legging sport compression",
    "Brassière sport maintien",
    "Veste de sport coupe-vent",
    "Pantalon jogging confortable",
    "Débardeur fitness respirant",
    "Chaussettes sport techniques",
    
    # Children's clothing
    "T-shirt enfant motif animal",
    "Robe petite fille princesse",
    "Pantalon enfant résistant",
    "Pyjama enfant doux",
    "Manteau enfant chaud",
    "Chaussures enfant premiers pas",
    
    # Underwear and lingerie
    "Soutien-gorge confort dentelle",
    "Culotte coton bio",
    "Boxer homme coton",
    "Chaussettes coton respirantes",
    "Collants fins transparents",
    
    # Outerwear
    "Manteau d'hiver long",
    "Parka imperméable capuche",
    "Doudoune légère compactable",
    "Trench-coat classique beige",
    "Imperméable pluie jaune",
    
    # Sleepwear
    "Pyjama satin luxueux",
    "Chemise de nuit coton",
    "Peignoir éponge doux",
    "Chaussons maison confort"
]

# Categories for better organization
product_categories = {
    "vêtements_femme": products[0:10],
    "vêtements_homme": products[10:20],
    "chaussures": products[20:30],
    "accessoires": products[30:40],
    "sport": products[40:48],
    "enfant": products[48:54],
    "sous_vêtements": products[54:59],
    "extérieur": products[59:64],
    "nuit": products[64:68]
}

# Pre-compute embeddings to optimize performance
logger.info("Computing product embeddings...")
product_embeddings = model.encode(products)
logger.info(f"Embeddings computed successfully for {len(products)} products!")

class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    threshold: Optional[float] = Field(SIMILARITY_THRESHOLD, description="Minimum similarity threshold", ge=0.0, le=1.0)
    max_results: Optional[int] = Field(MAX_RESULTS, description="Maximum number of results", ge=1, le=50)
    category_filter: Optional[str] = Field(None, description="Filter by category")

class ProductResponse(BaseModel):
    product: str
    score: float
    relevance: str
    category: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    threshold_used: float
    total_matches: int
    results: List[ProductResponse]
    has_relevant_results: bool
    categories_found: List[str]

def get_relevance_label(score: float) -> str:
    """Converts score to relevance label"""
    if score >= 0.7:
        return "Très pertinent"
    elif score >= 0.5:
        return "Pertinent"
    elif score >= 0.3:
        return "Moyennement pertinent"
    elif score >= 0.2:
        return "Faiblement pertinent"
    else:
        return "Peu pertinent"

def get_product_category(product: str) -> Optional[str]:
    """Finds the category of a product"""
    for category, products_in_category in product_categories.items():
        if product in products_in_category:
            return category
    return None

@app.get("/")
async def root():
    return {
        "message": "Semantic Search API v2.0", 
        "status": "active",
        "total_products": len(products),
        "categories": list(product_categories.keys()),
        "model": "all-MiniLM-L6-v2"
    }

@app.post("/search", response_model=SearchResponse)
async def semantic_search(search_query: SearchQuery):
    """
    Enhanced semantic search with threshold and category filtering
    """
    try:
        # Query validation
        if not search_query.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
        
        logger.info(f"Recherche pour: '{search_query.query}' avec seuil {search_query.threshold}")
        
        # Category filtering if specified
        search_products = products
        if search_query.category_filter:
            if search_query.category_filter in product_categories:
                search_products = product_categories[search_query.category_filter]
                # Recalculate embeddings for filtered category
                search_embeddings = model.encode(search_products)
            else:
                raise HTTPException(status_code=400, detail=f"Catégorie '{search_query.category_filter}' non trouvée")
        else:
            search_embeddings = product_embeddings
        
        # Generate query embedding
        query_embedding = model.encode([search_query.query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, search_embeddings)[0]
        
        # Create results with scores and filter by threshold
        results = []
        categories_found = set()
        
        for i, (product, score) in enumerate(zip(search_products, similarities)):
            if score >= search_query.threshold:
                category = get_product_category(product)
                if category:
                    categories_found.add(category)
                
                results.append(ProductResponse(
                    product=product,
                    score=float(score),
                    relevance=get_relevance_label(score),
                    category=category
                ))
        
        # Sort by descending score
        results = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Limit results
        results = results[:search_query.max_results]
        
        response = SearchResponse(
            query=search_query.query,
            threshold_used=search_query.threshold,
            total_matches=len(results),
            results=results,
            has_relevant_results=len(results) > 0,
            categories_found=list(categories_found)
        )
        
        logger.info(f"Trouvé {len(results)} résultats pour '{search_query.query}'")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/analyze-similarity")
async def analyze_similarity(search_query: SearchQuery):
    """
    Endpoint to analyze similarity score distribution
    """
    try:
        if not search_query.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
        
        query_embedding = model.encode([search_query.query])
        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        
        # Score statistics
        stats = {
            "query": search_query.query,
            "statistics": {
                "min_score": float(np.min(similarities)),
                "max_score": float(np.max(similarities)),
                "mean_score": float(np.mean(similarities)),
                "std_score": float(np.std(similarities)),
                "median_score": float(np.median(similarities))
            },
            "distribution": {
                "très_pertinent (≥0.7)": int(np.sum(similarities >= 0.7)),
                "pertinent (0.5-0.7)": int(np.sum((similarities >= 0.5) & (similarities < 0.7))),
                "moyennement_pertinent (0.3-0.5)": int(np.sum((similarities >= 0.3) & (similarities < 0.5))),
                "faiblement_pertinent (0.2-0.3)": int(np.sum((similarities >= 0.2) & (similarities < 0.3))),
                "peu_pertinent (<0.2)": int(np.sum(similarities < 0.2))
            },
            "top_scores": [
                {"product": product, "score": float(score), "relevance": get_relevance_label(score)}
                for product, score in sorted(zip(products, similarities), key=lambda x: x[1], reverse=True)[:10]
            ],
            "recommended_threshold": float(np.percentile(similarities, 70))  # 70th percentile
        }
        
        return stats
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.get("/products")
async def get_all_products():
    """Endpoint to retrieve all products with their categories"""
    return {
        "total_products": len(products),
        "products": products,
        "categories": product_categories,
        "products_by_category": {
            category: len(products_list) 
            for category, products_list in product_categories.items()
        }
    }

@app.get("/categories")
async def get_categories():
    """Endpoint to retrieve all available categories"""
    return {
        "categories": list(product_categories.keys()),
        "category_details": {
            category: {
                "name": category,
                "product_count": len(products_list),
                "sample_products": products_list[:3]
            }
            for category, products_list in product_categories.items()
        }
    }

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "embeddings_computed": product_embeddings is not None,
        "total_products": len(products),
        "api_version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)