import pandas as pd

def safe_label_encode(encoder, values):
    """Encode values but map unseen labels to -1."""
    known = set(encoder.classes_)
    output = []

    for v in values:
        if v in known:
            output.append(encoder.transform([v])[0])
        else:
            output.append(-1)   # unseen product/store ID
    return output


def preprocess_input(df, le_prod, le_store, features):

    # Encode Product ID safely
    if "Product ID" in df.columns:
        df["product_id_enc"] = safe_label_encode(le_prod, df["Product ID"])
        df["store_id_enc"]   = safe_label_encode(le_store, df["Store ID"])
        df = df.drop(["Product ID", "Store ID"], axis=1)

    # Ensure all features exist
    for col in features:
        if col not in df.columns:
            df[col] = 0      # Add missing columns safely

    df = df[features]
    return df
