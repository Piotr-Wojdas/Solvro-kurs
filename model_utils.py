from sklearn.metrics import precision_score, recall_score

class ModelTester:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def test_model(self, model):
        model.fit(self.x_train, self.y_train)

        y_pred = model.predict(self.x_test)

        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        print(f"--- Model: {type(model).__name__} ---")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print("-" * 30)

        return (precision, recall)

