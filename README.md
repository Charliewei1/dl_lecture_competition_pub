# DL基礎講座2024　最終課題「Visual Question Answering（VQA）」

## VizWiz(2023 edition) dataset [[link](https://www.kaggle.com/datasets/nqa112/vizwiz-2023-edition)] の詳細
- 24842枚の画像があり，訓練データには各画像に対して1つの質問と10人の回答者による回答．
  - 10人の回答はすべて同じであるとは限らない．
- 24842のデータの内，80%(19873)が訓練データ，20%(4969)がテストデータとして与えられる．
  - テストデータに対する回答は正解ラベルとし，訓練時には与えられない．
  - データ提供元ではtrainとvalに分かれているが，データの配布前に運営の方でtrainとvalをランダムに分け直している．

## タスクの詳細
- 本コンペでは，与えられた画像と質問に対し，適切な回答をモデルに出力させる．
- 評価は[VQA](https://visualqa.org/index.html)(Visual Question Answering)に基づき，以下の式で計算される．
$$\text{Acc}(ans) = min(\frac{humans \ that \ said \ ans}{3}, 1)$$
- 1つのデータに対し，10人の回答の内9人の回答を選択し上記の式で性能を評価した，10パターンのAccの平均をそのデータに対するAccとする．
- 予測結果と正解ラベルを比較する前に，回答をlowercaseにする，冠詞は削除するなどの前処理を行う（[詳細](https://visualqa.org/evaluation.html)）．
