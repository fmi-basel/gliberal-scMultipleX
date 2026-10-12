import pickle


def classify_me(df, classifier_path, unique_object_name):

    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)

    new_prediction = clf.predict(df)

    new_prediction = new_prediction.reset_index(level="label", drop=True).to_frame()

    # merge by index
    # check that column name exists for merging
    if unique_object_name in df.columns:
        df = df.set_index(unique_object_name)
    else:
        raise ValueError("%s must be a column in dataframe" % unique_object_name)

    # index of df must match index of classifier output
    df_predicted = df.join(new_prediction)

    # reset index so that original df indexing remains unchanged
    df_predicted = df_predicted.reset_index(drop=False)
    df_predicted

    return df_predicted, new_prediction, clf._class_names
