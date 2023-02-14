import json
import pandas as pd

# modelop.init
def init():
    global BINS
    global BUCKET_COL
    global LABEL_COL
    global POSITIVE_LABEL

	# job = json.loads(init_param["rawJson"])
    BINS = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    BUCKET_COL= "ally_score"
    LABEL_COL = "defaulted"
    POSITIVE_LABEL = 1

# modelop.metrics
def metrics(
        data: pd.DataFrame, 
        bins: list, 
        bucket_column: str, 
        label_column: str, 
        positive_label: str) -> str:
    bucketed_data = data.groupby([label_column, pd.cut(data[bucket_column], bins)]).size().unstack().T
    bucketed_data['percent'] =  (bucketed_data[positive_label] / data.shape[0])
    list_of_values = []
    for i, row in bucketed_data.iterrows():
        values = {}
        values[f'{bucket_column}_bucket'] = str(i)
        values['percent'] = row['percent']
        list_of_values.append({'values': values})
    return {'Rank_Order': list_of_values}

def main():
    init()
    data = pd.read_csv('./rob_test.csv')
    result = metrics(
            data,
            BINS,
            BUCKET_COL,
            LABEL_COL,
            POSITIVE_LABEL
        )
    print(json.dumps(result, indent=3, sort_keys=True))

if __name__ == '__main__':
	main()