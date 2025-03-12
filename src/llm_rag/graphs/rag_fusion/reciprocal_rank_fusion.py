from itertools import chain


def result_func(q):
    return database[q]


def rank_func(results, d):
    return results.index(d) + 1


def reciprocal_rank_fusion(queries, d, k, result_func, rank_func):
    return sum(
        [
            1.0 / (k + rank_func(result_func(q), d)) if d in result_func(q) else 0
            for q in queries
        ]
    )


if __name__ == "__main__":
    database = {
        "query1": ["doc1", "doc2", "doc3"],
        "query2": ["doc3", "doc1", "doc2"],
    }
    queries = list(database.keys())
    documents = set(chain.from_iterable(database.values()))
    k = 60

    results = {
        document: reciprocal_rank_fusion(queries, document, k, result_func, rank_func)
        for document in documents
    }
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(sorted_results)
