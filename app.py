from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import re
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ── Groq setup ────────────────────────────────────────────────
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"
groq_client  = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

# ── Load data ─────────────────────────────────────────────────
trending_products = pd.read_csv("models/trending_products.csv")
train_data        = pd.read_csv("models/clean_data.csv")

train_data['ImageURL'] = train_data['ImageURL'].astype(str).apply(
    lambda x: x.split('|')[0].strip()
)
train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce').fillna(0)
train_data['Rating'] = train_data['Rating'].apply(
    lambda x: 0 if x == -2147483648 else x
)
train_data['ReviewCount'] = pd.to_numeric(train_data['ReviewCount'], errors='coerce').fillna(0)
train_data['ReviewCount'] = train_data['ReviewCount'].apply(
    lambda x: 0 if x == -2147483648 else x
)

# ── DB ────────────────────────────────────────────────────────
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom?charset=utf8mb4"
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'connect_args': {'charset': 'utf8mb4'}
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Signup(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email    = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# ── Constants ─────────────────────────────────────────────────
random_image_urls = [
    "static/img/img1.jpg", "static/img/img2.jpg",
    "static/img/img3.jpg", "static/img/img4.jpg",
    "static/img/img5.jpg", "static/img/img6.jpg",
    "static/img/img7.jpg", "static/img/img8.jpg",
]
price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]


# ── Utility functions ─────────────────────────────────────────
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text


def _resolve_image(url_val):
    """Always returns a usable image URL."""
    url = str(url_val).strip()
    if url.startswith('http://') or url.startswith('https://'):
        return url
    if url.startswith('static/'):
        return '/' + url
    if url.startswith('/static/'):
        return url
    return 'https://placehold.co/300x220/f5a623/ffffff?text=Product'


def _df_to_products(df, limit=6):
    """Convert dataframe rows to product dicts for the chat widget."""
    result = []
    for _, row in df.head(limit).iterrows():
        try:
            r = float(row.get('Rating', 0))
            rating_str = str(round(r, 1)) if r > 0 else 'N/A'
        except Exception:
            rating_str = 'N/A'
        result.append({
            'name':   str(row.get('Name',  'Product')),
            'brand':  str(row.get('Brand', 'N/A')),
            'image':  _resolve_image(row.get('ImageURL', '')),
            'rating': rating_str,
            'price':  random.choice(price)
        })
    return result


def session_vars():
    cart = session.get('cart', [])
    return {
        'logged_in':  session.get('logged_in', False),
        'username':   session.get('username', ''),
        'cart_count': sum(i['quantity'] for i in cart)
    }


# ── Recommendation engine ─────────────────────────────────────
def content_based_recommendations(train_data, item_name, top_n=8):
    if 'Name' not in train_data.columns:
        return pd.DataFrame()
    matches = train_data[
        train_data['Name'].astype(str).str.lower().str.contains(item_name.lower(), na=False)
    ]
    if matches.empty:
        return pd.DataFrame()

    matched_name = matches.iloc[0]['Name']
    tfidf_data = train_data.copy()
    tfidf_data['Tags'] = tfidf_data['Tags'].fillna('') + " " + tfidf_data['Name'].fillna('')

    tfidf_vectorizer     = TfidfVectorizer(stop_words='english')
    tfidf_matrix         = tfidf_vectorizer.fit_transform(tfidf_data['Tags'])
    cosine_sim           = cosine_similarity(tfidf_matrix, tfidf_matrix)

    match_mask = tfidf_data['Name'].astype(str).str.lower().str.contains(matched_name.lower(), na=False)
    if not match_mask.any():
        return pd.DataFrame()

    item_pos     = tfidf_data.index.get_loc(tfidf_data[match_mask].index[0])
    similar      = sorted(enumerate(cosine_sim[item_pos]), key=lambda x: x[1], reverse=True)
    top_indices  = [x[0] for x in similar[1:top_n + 1]]
    return tfidf_data.iloc[top_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]


def collaborative_filtering_recommendations(train_data, target_product_name, top_n=4):
    try:
        df = train_data.copy()
        df['Name']        = df['Name'].fillna('')
        df['Tags']        = df['Tags'].fillna('')
        df['Brand']       = df['Brand'].fillna('Unknown')
        df['Rating']      = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
        df['ReviewCount'] = pd.to_numeric(df['ReviewCount'], errors='coerce').fillna(0)

        cat_col = next((c for c in ['Category','category','ProductType','Type'] if c in df.columns), None)
        matches = df[df['Name'].str.lower().str.contains(target_product_name.lower(), na=False)]
        if matches.empty:
            return pd.DataFrame()

        seed_row  = matches.iloc[0]
        seed_name = seed_row['Name']
        seed_cat  = seed_row[cat_col] if cat_col else None

        df['combined'] = df['Tags'] + " " + df['Name']
        tfidf_matrix   = TfidfVectorizer(stop_words='english', max_features=8000).fit_transform(df['combined'])
        seed_mask      = df['Name'].str.lower().str.contains(seed_name.lower(), na=False)
        seed_loc       = df.index.get_loc(df[seed_mask].index[0])
        content_scores = cosine_similarity(tfidf_matrix[seed_loc], tfidf_matrix).flatten()
        collab_scores  = np.zeros(len(df))

        try:
            cat_df = df[df[cat_col] == seed_cat].copy() if cat_col and seed_cat \
                     else df.sample(min(8000, len(df)), random_state=42).copy()
            if len(cat_df) > 1:
                cat_df['RatingBucket'] = pd.cut(
                    cat_df['Rating'], bins=[0,1,2,3,4,5],
                    labels=['r1','r2','r3','r4','r5'], include_lowest=True
                ).astype(str)
                cat_df = cat_df.drop_duplicates(subset=['Name'])
                pivot  = cat_df.pivot_table(index='Name', columns='RatingBucket',
                                            values='ReviewCount', aggfunc='sum', fill_value=0)
                if seed_name in pivot.index and len(pivot) > 1:
                    sim_row     = cosine_similarity(pivot)[pivot.index.tolist().index(seed_name)]
                    name_to_sim = dict(zip(pivot.index.tolist(), sim_row))
                    collab_scores = np.array([name_to_sim.get(n, 0.0) for n in df['Name']])
        except Exception:
            pass

        c_max        = collab_scores.max()
        collab_norm  = collab_scores  / (c_max + 1e-9)
        content_norm = content_scores / (content_scores.max() + 1e-9)
        blended      = (0.60 * collab_norm + 0.40 * content_norm) if c_max > 0.01 \
                       else (0.55 * content_norm + 0.45 * df['Rating'].values / (df['Rating'].max() + 1e-9))
        blended[seed_loc] = -1

        seen, final = set(), []
        for idx in np.argsort(blended)[::-1][:top_n * 4]:
            name = df.iloc[idx]['Name']
            if name not in seen and name.lower() != seed_name.lower():
                seen.add(name); final.append(idx)
            if len(final) >= top_n:
                break

        return df.iloc[final][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']] if final \
               else pd.DataFrame()
    except Exception as e:
        print(f"Collab error: {e}")
        return pd.DataFrame()


# ── Build catalog context for Groq (once at startup) ──────────
def _build_catalog():
    try:
        brands   = train_data['Brand'].dropna().unique().tolist()[:80]
        samples  = train_data['Name'].dropna().sample(min(50, len(train_data)), random_state=1).tolist()
        trending = trending_products['Name'].dropna().tolist()[:10]
        return (
            f"Brands: {', '.join(str(b) for b in brands)}.\n"
            f"Sample products: {', '.join(str(n) for n in samples)}.\n"
            f"Trending: {', '.join(str(n) for n in trending)}."
        )
    except Exception:
        return "Beauty and cosmetics ecommerce store."

CATALOG = _build_catalog()

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are ShopBot, a smart shopping assistant for a beauty and cosmetics store.

Rules:
- Be warm, brief and helpful. Max 2 sentences for your reply text.

- When the user wants products, you MUST output a SEARCH_KEYWORD line at the end.
- SEARCH_KEYWORD tells the backend what to search. Be specific — use the actual product type.
- For greetings/casual chat, do NOT output SEARCH_KEYWORD.
- Never say you can't help with shopping queries — always extract a keyword.

Store catalog:
{CATALOG}

SEARCH_KEYWORD rules:
- User wants lipstick → SEARCH_KEYWORD: lipstick
- User wants something for dry skin → SEARCH_KEYWORD: moisturizer
- User wants OPI brand → SEARCH_KEYWORD: OPI
- User wants trending/popular → SEARCH_KEYWORD: TRENDING
- User wants top rated → SEARCH_KEYWORD: TOP_RATED
- User says hi/thanks/bye → no SEARCH_KEYWORD

Format: always put SEARCH_KEYWORD: <word> on its own line at the very end.
"""


# ── Fetch products by keyword ─────────────────────────────────
def _fetch_products(keyword, logged_in):
    """Run the recommendation engine based on keyword from Groq."""
    keyword = keyword.strip()

    if keyword == 'TRENDING':
        products = []
        for _, row in trending_products.head(6).iterrows():
            products.append({
                'name':   str(row.get('Name', 'Product')),
                'brand':  str(row.get('Brand', 'N/A')),
                'image':  _resolve_image(row.get('ImageURL', '')),
                'rating': str(row.get('Rating', 'N/A')),
                'price':  random.choice(price)
            })
        return products

    if keyword == 'TOP_RATED':
        recs = train_data.sort_values('Rating', ascending=False).head(6)
        return _df_to_products(recs)

    # Content-based search
    recs = content_based_recommendations(train_data, keyword, top_n=6)

    # Add collab recs for logged-in users
    if logged_in and not recs.empty:
        collab = collaborative_filtering_recommendations(train_data, keyword, top_n=3)
        if not collab.empty:
            existing = set(recs['Name'].tolist())
            collab   = collab[~collab['Name'].isin(existing)]
            recs     = pd.concat([recs, collab]).head(8)

    if not recs.empty:
        return _df_to_products(recs)

    # Fallback: try brand search
    brand_results = train_data[
        train_data['Brand'].astype(str).str.lower().str.contains(keyword.lower(), na=False)
    ].head(6)
    if not brand_results.empty:
        return _df_to_products(brand_results)

    # Fallback: partial name match
    name_results = train_data[
        train_data['Name'].astype(str).str.lower().str.contains(keyword.lower(), na=False)
    ].head(6)
    if not name_results.empty:
        return _df_to_products(name_results)

    return []


# ═══════════════════════════════════════════════════════════════
#  CHAT ROUTE
# ═══════════════════════════════════════════════════════════════
@app.route('/chat', methods=['POST'])
def chat():
    data     = request.get_json()
    user_msg = (data.get('message') or '').strip()

    if not user_msg:
        return jsonify({'reply': 'Say something and I\'ll help! 🛍️', 'products': []})

    username  = session.get('username', '')
    logged_in = session.get('logged_in', False)

    # Maintain last 6 turns for context
    history = session.get('chat_history', [])
    history.append({"role": "user", "content": user_msg})
    if len(history) > 6:
        history = history[-6:]

    products = []
    reply    = ''

    try:
        system = SYSTEM_PROMPT
        if username:
            system += f"\n\nUser's name is {username}. Address them by name once when greeting."

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system}] + history,
            temperature=0.5,
            max_tokens=250,
        )

        full_text = response.choices[0].message.content.strip()
        print(f"[Groq raw]: {full_text}")  # debug — visible in Flask console

        # Extract SEARCH_KEYWORD if present
        search_keyword = None
        if 'SEARCH_KEYWORD:' in full_text:
            parts          = full_text.split('SEARCH_KEYWORD:')
            reply          = parts[0].strip()
            search_keyword = parts[1].strip().split('\n')[0].strip()
        else:
            reply = full_text

        # Fetch products
        if search_keyword:
            products = _fetch_products(search_keyword, logged_in)
            # If Groq gave a keyword but no products found, say so clearly
            if not products:
                reply += f"\n\nCouldn't find exact matches for '{search_keyword}'. Try a different term!"

        # Save history
        history.append({"role": "assistant", "content": full_text})
        session['chat_history'] = history[-6:]
        session.modified = True

    except Exception as e:
        # Print the REAL error in console so you can debug
        print(f"[Groq ERROR]: {type(e).__name__}: {e}")
        reply    = f"Error: {type(e).__name__} — {str(e)[:120]}"
        products = []

    return jsonify({'reply': reply, 'products': products})


# ═══════════════════════════════════════════════════════════════
#  CART ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    data      = request.get_json()
    name      = (data.get('name') or '').strip()
    brand     = data.get('brand', 'N/A')
    image     = data.get('image', '')
    rating    = data.get('rating', 'N/A')
    price_val = data.get('price', 50)

    if not name:
        return jsonify({'success': False, 'message': 'Invalid product'}), 400

    cart = session.get('cart', [])
    for item in cart:
        if item['name'] == name:
            item['quantity'] += 1
            session['cart'] = cart; session.modified = True
            return jsonify({'success': True,
                            'cart_count': sum(i['quantity'] for i in cart),
                            'message': 'Quantity updated!'})

    cart.append({'name': name, 'brand': brand, 'image': image,
                 'rating': rating, 'price': price_val, 'quantity': 1})
    session['cart'] = cart; session.modified = True
    return jsonify({'success': True,
                    'cart_count': sum(i['quantity'] for i in cart),
                    'message': f'"{truncate(name, 25)}" added to cart!'})


@app.route('/update_cart', methods=['POST'])
def update_cart():
    data   = request.get_json()
    name   = data.get('name', '')
    action = data.get('action', '')
    cart   = session.get('cart', [])
    for item in cart:
        if item['name'] == name:
            if action == 'increase':   item['quantity'] += 1
            elif action == 'decrease':
                item['quantity'] -= 1
                if item['quantity'] <= 0: cart.remove(item)
            break
    session['cart'] = cart; session.modified = True
    total = sum(i['price'] * i['quantity'] for i in cart)
    return jsonify({'success': True,
                    'cart_count': sum(i['quantity'] for i in cart),
                    'total': round(total, 2)})


@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    data = request.get_json()
    cart = [i for i in session.get('cart', []) if i['name'] != data.get('name', '')]
    session['cart'] = cart; session.modified = True
    total = sum(i['price'] * i['quantity'] for i in cart)
    return jsonify({'success': True,
                    'cart_count': sum(i['quantity'] for i in cart),
                    'total': round(total, 2)})


@app.route('/cart')
def cart_page():
    cart  = session.get('cart', [])
    total = sum(i['price'] * i['quantity'] for i in cart)
    return render_template('cart.html', cart=cart, total=round(total, 2), **session_vars())


# ═══════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════
def _imgs():
    return [random_image_urls[i % len(random_image_urls)] for i in range(len(trending_products))]


@app.route("/")
def index():
    return render_template('index.html',
                           trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=_imgs(),
                           random_price=random.choice(price),
                           **session_vars())


@app.route("/main")
def main():
    return render_template('main.html',
                           content_based_rec=pd.DataFrame(),
                           collab_rec=pd.DataFrame(),
                           **session_vars())


@app.route("/index")
def indexredirect():
    return render_template('index.html',
                           trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=_imgs(),
                           random_price=random.choice(price),
                           **session_vars())


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']
        if Signup.query.filter_by(username=username).first():
            return render_template('index.html',
                                   trending_products=trending_products.head(8),
                                   truncate=truncate,
                                   random_product_image_urls=_imgs(),
                                   random_price=random.choice(price),
                                   signup_message='Username already exists! Please Sign In.',
                                   signup_error=True, open_signin=True, **session_vars())
        db.session.add(Signup(username=username, email=email, password=password))
        db.session.commit()
        return render_template('index.html',
                               trending_products=trending_products.head(8),
                               truncate=truncate,
                               random_product_image_urls=_imgs(),
                               random_price=random.choice(price),
                               signup_message='Account created! Please Sign In.',
                               open_signin=True, **session_vars())
    return redirect(url_for('index'))


@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        user     = Signup.query.filter_by(username=username, password=password).first()
        if user:
            session['logged_in'] = True
            session['username']  = username
            return render_template('index.html',
                                   trending_products=trending_products.head(8),
                                   truncate=truncate,
                                   random_product_image_urls=_imgs(),
                                   random_price=random.choice(price),
                                   signup_message=f'Welcome back, {username}!',
                                   logged_in=True, username=username,
                                   cart_count=sum(i['quantity'] for i in session.get('cart', [])))
        user_exists = Signup.query.filter_by(username=username).first()
        msg = 'Account not found! Please Sign Up.' if not user_exists else 'Incorrect password!'
        return render_template('index.html',
                               trending_products=trending_products.head(8),
                               truncate=truncate,
                               random_product_image_urls=_imgs(),
                               random_price=random.choice(price),
                               signup_message=msg,
                               signin_error=True, open_signin=True, **session_vars())
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = (request.form.get('prod') or '').strip()
        if not prod:
            return render_template('main.html', message="Please enter a product name.",
                                   content_based_rec=pd.DataFrame(),
                                   collab_rec=pd.DataFrame(), **session_vars())

        content_based_rec = content_based_recommendations(train_data, prod, top_n=6)
        collab_rec        = pd.DataFrame()

        if session.get('logged_in'):
            collab_rec = collaborative_filtering_recommendations(train_data, prod, top_n=4)
            if not content_based_rec.empty and not collab_rec.empty:
                content_names = set(content_based_rec['Name'].tolist())
                collab_rec    = collab_rec[~collab_rec['Name'].isin(content_names)]

        if content_based_rec.empty:
            return render_template('main.html',
                                   message=f"No results for '{prod}'. Try another keyword.",
                                   content_based_rec=pd.DataFrame(),
                                   collab_rec=pd.DataFrame(), **session_vars())

        return render_template('main.html',
                               content_based_rec=content_based_rec,
                               collab_rec=collab_rec,
                               truncate=truncate,
                               random_price=random.choice(price),
                               **session_vars())

    return render_template('main.html',
                           content_based_rec=pd.DataFrame(),
                           collab_rec=pd.DataFrame(), **session_vars())

@app.route('/payment')
def payment():
    cart = session.get('cart', [])
    total = sum(i['price'] * i['quantity'] for i in cart)
    return render_template('payment.html', total=int(total * 1.08), **session_vars())

if __name__ == "__main__":
    app.run(debug=True)