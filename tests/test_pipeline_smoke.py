from src.train import build_pipeline


def test_pipeline_smoke():
    pipe = build_pipeline()
    X = ["hello how are you","win money now","free tickets","call me later"]
    y = [0,1,1,0]
    pipe.fit(X,y)
    preds = pipe.predict(["free prize now","see you soon"])
    assert len(preds) == 2

