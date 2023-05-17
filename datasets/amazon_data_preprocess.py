import os.path

import numpy
import pandas as pd
import json
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
import warnings
from typing import Dict
from datasets import load_small_dataset
from tqdm import tqdm
import torch
from torchtext.vocab import GloVe
from utils import *
import random
from sklearn.preprocessing import StandardScaler
import nltk
import copy

warnings.filterwarnings("ignore")


def get_merge_category(num_class, cate_mode, dataset_name):
    if dataset_name == 'app':
        if cate_mode == 0:
            merge_conditions = [  # do not modity the order!!!
                ['oven', 'burner', 'range', 'hood', 'cooktop', 'wave', 'warming drawer', 'grill'],
                ['washer', 'dryer', 'laundry', 'bin'],
                ['travel', 'car', 'motor'],
                ['refrigerator', 'freezer', 'frigidaire'],
                ['satin', 'trim', 'wool', 'soft', 'lingerie', 'breathable', 'pocket', 'closure'],
                ['trash', 'garbage', 'waste'],
                ['humid', 'air', 'moisture'],
                ['usb', 'fastener', 'plug'],
                ['gift', 'friend', 'bride', 'bridal', 'engagement', 'bachelorette'],
                ['door', 'filter', 'drain', 'vent', 'blower', 'appliances']
            ]
        elif cate_mode == 1:
            merge_conditions = [  # do not modity the order!!!
                ['oven', 'burner', 'range', 'hood', 'cooktop', 'wave', 'warming drawer', 'grill'],
                ['washer', 'dryer', 'laundry'],
                ['travel', 'car', 'motor'],
                ['refrigerator', 'freezer', 'frigidaire'],
                ['satin', 'trim', 'wool', 'soft', 'lingerie', 'breathable', 'pocket', 'closure', 'leather', 'moisture'],
                ['trash', 'garbage', 'waste', 'bins'],
                ['humid'],
                ['gift', 'friend', 'bride', 'bridal', 'engagement', 'bachelorette'],
                ['filter', 'drain', 'vent', 'blower', 'absorption', 'pump'],
                ['appliances']
            ]
    elif dataset_name == 'movies':
        merge_conditions = [  # do not modity the order !!!
            ['oven', 'burner', 'range', 'hood', 'cooktop', 'wave', 'warming drawer', 'grill'],
            ['washer', 'dryer', 'laundry'],
            ['travel', 'car', 'motor'],
            ['refrigerator', 'freezer', 'frigidaire'],
            ['satin', 'trim', 'wool', 'soft', 'lingerie', 'breathable', 'pocket', 'closure', 'leather', 'moisture'],
            ['trash', 'garbage', 'waste', 'bins'],
            ['humid'],
            ['gift', 'friend', 'bride', 'bridal', 'engagement', 'bachelorette'],
            ['filter', 'drain', 'vent', 'blower', 'absorption', 'pump'],
            ['appliances']
        ]
    elif dataset_name == 'sftw':
        merge_conditions = [  # do not modity the order!!!
            ['books', 'comics', 'lifestyle'],
            ['communication', 'chat'],
            ['education', 'learning'],
            ['finance', 'tax'],
            ['games'],
            ['health', 'fitness'],
            ['kids', 'children'],
            ['music', 'podcast'],
            ['news', 'magazines'],
            ['photography'],
            ['productivity', 'business', 'design', 'editing', 'dvd viewing', 'development', 'composit', 'video'],
            ['ringtones'],
            ['travel', 'navigation'],
            ['weather'],
            ['linux', 'operating system', 'unix'],
            ['server'],
            ['security', 'antivirus', 'backup', 'digital software', 'pc maintenance', 'utilities', 'firewall']
        ]
    return merge_conditions


def count_label(label):
    label = label[np.where(label >= 0)]
    print("None zero", len(label))
    print(f"Max-label:{np.max(label)} Min-label:{np.min(label)} Average-label:{np.mean(label)}")
    y, x = np.histogram(label, np.arange(np.min(label), np.max(label) + 2), density=True)
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y)
    ax.set(xlabel='label', ylabel='nodes', title='Label Histogram')
    fig.savefig("Label Histogram.png")
    plt.show()


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def count_degrees(graph: dgl.DGLGraph):
    # for observe graph data
    degrees = ((graph.in_degrees() + graph.out_degrees()) / 2).numpy()
    print(f"Max-degree:{np.max(degrees)} Min-degree:{np.min(degrees)} Average-degree:{np.mean(degrees)}")
    degrees[degrees > 10] = 10
    y, x = np.histogram(degrees, np.arange(np.min(degrees), np.max(degrees) + 2), density=True)
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y)
    ax.set(xlabel='degree', ylabel='nodes', title='Degree Histogram')
    fig.savefig("degree.png")
    plt.show()


class amazon_graph_info():
    def __init__(
            self,
            dataset_name,
            filter_degree=0,
            feature_dim=300,
            save_path=None,
            preprocess_path=None,
            review_path=None,
            meta_path=None,
            duplicate_path=None,
            graph_type='user_product',
            user_feature_type='description',
            product_feature_type='description'
    ):
        self.user_feature_type = user_feature_type
        self.product_feature_type = product_feature_type
        self.co_product_edges = None
        self.review_path = None
        self.user_nodes_feature = None
        self.product_nodes_feature = None
        self.id2users = None
        self.users_dict = None
        self.id2products = None
        self.products_dict = None
        self.num_products = None
        self.num_users = None
        self.dataset_name = dataset_name
        if 'user_product' in graph_type:
            self.load_info_user_product(
                filter_degree=filter_degree,
                feature_dim=feature_dim,
                save_path=save_path,
                preprocess_path=preprocess_path,
                review_path=review_path,
                meta_path=meta_path,
                duplicate_path=duplicate_path
            )
        elif 'co_product' in graph_type:
            self.load_info_co_product(
                filter_degree=filter_degree,
                feature_dim=feature_dim,
                save_path=save_path,
                preprocess_path=preprocess_path,
                review_path=review_path,
                meta_path=meta_path,
                duplicate_path=duplicate_path,
                product_feature_type=product_feature_type
            )
        return

    def load_review_data(self, path):
        self.review_path = path
        # build graph (bidirected, no nodes features, only edges ratings) from json file
        # 0 - num_users id are user nodes
        # num_users - (num_users + num_products) id are product nodes
        with open(path) as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line))
        print(f"Total reviews: {len(data)}")
        keys = []
        for line in data:
            keys += line.keys()
        keys = list(set(keys))
        key_count = {}
        products = []
        users = []
        ratings = []
        times = []
        reviews = []
        for line in data:
            for key in keys:
                if key in line.keys():
                    if key in line.keys():
                        if isinstance(line[key], (float, int, bool)) or (
                                line[key] is not None and line[key] not in ['$', '$$'] and len(line[key]) > 0):
                            key_count[key] = key_count.get(key, 0) + 1
                        else:
                            key_count[key] = key_count.get(key, 0)
            products.append(line['asin'])
            users.append(line['reviewerID'])
            ratings.append(line['overall'])
            times.append(line['unixReviewTime'])
            if 'reviewText' in line.keys():
                reviews.append(line['reviewText'])
            else:
                reviews.append("")
        print(key_count)
        products_id = list(set(products))
        users_id = list(set(users))
        self.num_products = len(products_id)
        self.num_users = len(users_id)
        print('num_products', self.num_products)
        print('num_users', self.num_users)
        users_dict = dict((users_id[i], i) for i in range(self.num_users))  # {118026874392842649478: 0 ...}
        users_id = dict((val, key) for key, val in users_dict.items())
        products_dict = dict(
            (products_id[i - self.num_users], i) for i in
            range(self.num_users, self.num_users + self.num_products))  # {0x37f42bff4edf7a43: 13 ...}
        products_id = dict((val, key) for key, val in products_dict.items())
        edges_user = [users_dict[i] for i in users]
        edges_product = [products_dict[i] for i in products]
        self.id2users = users_id
        self.users_dict = users_dict
        self.id2products = products_id
        self.products_dict = products_dict
        # merge parallel edges
        edges_dict = {}  # {(0,1233):{ratings:5.0, times:[time1,time2...]}...}
        for i, j, r, t, review in zip(edges_user, edges_product, ratings, times, reviews):
            edges_dict[(i, j)] = edges_dict.get((i, j), {'rating': 0, 'times': [], 'review': []})
            edges_dict[(i, j)]['rating'] += r
            edges_dict[(i, j)]['times'].append(t)
            edges_dict[(i, j)]['review'].append(review)
        # average the ratings
        for (i, j) in edges_dict.keys():
            edges_dict[(i, j)]['rating'] /= len(edges_dict[(i, j)]['times'])
            edges_dict[(i, j)]['review'] = " ".join(edges_dict[(i, j)]['review'])
        print(f'After merge paralled edges, edges changes from {len(edges_user)} to {len(edges_dict)}')

        self.edges_dict = edges_dict

    def load_meta_data(self, meta_data_path):
        self.meta_data_path = meta_data_path
        with open(meta_data_path) as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line))
        print(f"\nTotal metas: {len(data)}")
        keys = []
        for line in data:
            keys += line.keys()
        keys = list(set(keys))
        key_count = {}
        products = []
        names = []
        descriptions = []
        categories = []
        prices = []
        for line in data:
            for key in keys:
                if key in line.keys():
                    add = 1 if line[key] is not None and line[key] not in ['$', '$$'] and len(line[key]) > 0 else 0
                    key_count[key] = key_count.get(key, 0) + add
            if 'description' not in line.keys():
                line['description'] = ""
            if 'price' not in line.keys():
                line['price'] = ""
            if 'category' not in line.keys():
                line['category'] = []
            if 'title' not in line.keys():
                line['title'] = ""
            products.append(line['asin'])
            names.append(line['title'])
            descriptions.append(" ".join(line['description']))
            categories.append(line['category'] if len(line['category']) > 0 else None)
            prices.append(
                float(line['price'][1:].replace(",", "")) if len(line['price']) < 10 and len(
                    line['price']) > 0 else None)
        print(key_count)
        # generate catergory word bank and catergory label dict
        category_word_bank = []
        for entry in categories:
            if isinstance(entry, list):
                for category in entry:
                    category_word_bank.append(category)
        category_word_bank = list(set(category_word_bank))
        print(f'Different category labels: {len(category_word_bank)}')
        self.categories_dict = dict(
            (category_word_bank[i], i) for i in range(len(category_word_bank)))  # {"Park": 0 ...}
        self.id2catergories = dict((val, key) for key, val in self.categories_dict.items())  # {0: "Park" ...}
        # generate all meta-data products information dict
        products_info_dict = {}  # {B00005R7ZY:{'name':name,'description';des,'price';price}...}
        for product, name, des, price, cates in zip(products, names, descriptions, prices, categories):
            cate = [self.categories_dict[x] for x in cates] if cates is not None else None
            products_info_dict[product] = {'name': name, 'description': des, 'price': price, 'category': cate}
        print(f"Total products: {len(products_info_dict)}")
        self.products_info_dict = products_info_dict

    def count_users_price(self):
        id2products = self.id2products
        edges_dict = self.edges_dict
        products_info_dict = self.products_info_dict
        print("\n--- Generating user nodes features ---")
        user_prices = {}
        for (i, j) in edges_dict.keys():
            price = products_info_dict[id2products[j]]['price']
            price = price if price is not None else 0
            # user_key='description'
            if i not in user_prices.keys():
                user_prices[i] = [price]
            else:
                user_prices[i].append(price)
        user_prices = list(user_prices.values())
        for prices in user_prices:
            if np.std(prices, where=np.array(prices) > 0) > np.mean(prices, where=np.array(prices) > 0):
                print(prices)
                print(
                    f"mean: {np.mean(prices, where=np.array(prices) > 0)} std: {np.std(prices, where=np.array(prices) > 0)}")
        x = [np.mean(prices, where=np.array(prices) > 0) for prices in user_prices]
        y = [np.std(prices, where=np.array(prices) > 0) for prices in user_prices]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1)
        ax.plot([0, 200], [0, 200], c='r')
        ax.set(xlabel='mean price per user', ylabel='std price per user', title='User-Price Counts')
        fig.savefig("User-Price Counts.png")
        plt.show()

    def load_basic_info(
            self,
            filter_degree=0,
            save_path=None,
            preprocess_path=None,
            review_path=None,
            meta_path=None,
            duplicate_path=None,
    ):
        if preprocess_path is None or not os.path.exists(os.path.join(preprocess_path, "id2users.npy")):
            # build graph after preprocess data
            self.load_review_data(review_path)
            self.load_meta_data(meta_path)
            if duplicate_path is not None:
                self.assure_no_duplicated_products(duplicate_path)
            print("--- Building Graph ---")
            # has load meta_data, reassure that all the products are in meta_data, or delete them
            print("1. Cleaning products nodes that are not in meta or description is none")
            if hasattr(self, 'products_info_dict'):
                # clean products nodes that are not in meta or description is none
                product_reserve = []
                for j, product in self.id2products.items():
                    if product in self.products_info_dict.keys() \
                            and len(
                        nltk.word_tokenize(self.products_info_dict[product]['description'].replace("-", " "))) > 0:
                        product_reserve.append(j)
                self.update_index(self.id2users, self.id2products, self.edges_dict, self.id2users.keys(),
                                  product_reserve)
            # filter nodes whose degree is smaller than filter_degree
            print(f"2. Filtering nodes whose degree is smaller than {filter_degree}")
            id2users = self.id2users
            id2products = self.id2products
            edges_dict = self.edges_dict
            user_reserve, product_reserve = self.filter_nodes_with_degree(filter_degree)
            self.update_index(self.id2users, self.id2products, self.edges_dict, user_reserve, product_reserve)
            # need to filter new degree=0 nodes
            print(f"3. Filtering isolated nodes that may be caused by step 2")
            user_reserve, product_reserve = self.filter_nodes_with_degree(0)
            self.update_index(self.id2users, self.id2products, self.edges_dict, user_reserve, product_reserve)
            print(
                f"      filter ratio: users:{self.num_users}/{len(id2users)}, products:{self.num_products}/{len(id2products)}")
            print(
                f'      After filter degree {filter_degree}, edges changes from {len(edges_dict)} to {len(self.edges_dict)}')
            if save_path is not None:
                print(f"   Saving preprocess data into {save_path}")
                self.save_preprocess_info(save_path)

        else:  # load preprocessed data from npy files and build graph
            print(f"--- Building Graph from preprocessed data files in {preprocess_path} ---")
            self.load_preprocess_info(preprocess_path)
            print(f"Product Nodes    : {self.num_products}")
            print(f"User Nodes       : {self.num_users}")
            print(f"Undirected Edges : {len(self.edges_dict)}")

    def load_info_user_product(
            self,
            filter_degree=0,
            feature_dim=300,
            save_path=None,
            preprocess_path=None,
            review_path=None,
            meta_path=None,
            duplicate_path=None,
    ):
        self.load_basic_info(
            filter_degree=filter_degree,
            save_path=save_path,
            preprocess_path=preprocess_path,
            review_path=review_path,
            meta_path=meta_path,
            duplicate_path=duplicate_path,
        )
        print(
            f"4. Getting feature: User feature type: {self.user_feature_type};  Product feature type: {self.product_feature_type}")
        if self.user_nodes_feature is None:
            self.get_user_nodes_feature(feature_dim=feature_dim, feature_type=self.user_feature_type)
        if self.product_nodes_feature is None:
            self.get_product_nodes_feature(feature_dim=feature_dim, feature_type=self.product_feature_type)
        if save_path is not None:
            self.save_nodes_feature(save_path=save_path)

    def load_info_co_product(
            self,
            filter_degree=0,
            feature_dim=300,
            save_path=None,
            preprocess_path=None,
            review_path=None,
            meta_path=None,
            duplicate_path=None,
    ):
        self.load_basic_info(
            filter_degree=filter_degree,
            save_path=save_path,
            preprocess_path=preprocess_path,
            review_path=review_path,
            meta_path=meta_path,
            duplicate_path=duplicate_path,
        )
        print(f"4. Processing co-purchase info")
        user_products = {}
        for (i, j) in self.edges_dict.keys():
            if i not in user_products.keys():
                user_products[i] = [j]
            else:
                user_products[i].append(j)
        co_product_edges = []
        product_reserve = []
        for i, js in user_products.items():
            for j1 in js:
                for j2 in js:
                    if j1 != j2:
                        product_reserve += [j1, j2]
                        co_product_edges.append((j1, j2))
                        co_product_edges.append((j2, j1))
        product_reserve = list(set(product_reserve))
        self.update_index_co_purchase(self.id2users, self.id2products, co_product_edges, [], product_reserve)
        self.co_product_edges = [[j1, j2] for (j1, j2) in list(set(self.co_product_edges))]
        self.co_product_edges = torch.tensor(self.co_product_edges)
        print(f"Co-Product feature type: {self.product_feature_type}")
        if self.product_nodes_feature is None:
            self.get_product_nodes_feature(feature_dim=feature_dim, feature_type=self.product_feature_type)
        if save_path is not None:
            if not os.path.exists(os.path.join(save_path,
                                               f"product_feature_{self.product_feature_type}.npy")) and self.product_nodes_feature is not None:
                print(f"   Saving product feature ({self.product_feature_type}) into {save_path}")
                np.save(os.path.join(save_path, f"product_feature_{self.product_feature_type}.npy"),
                        self.product_nodes_feature)

    def build_graph_category_predict(
            self,
            split=(6, 2, 2),
            num_class=5,
            shuffle=True,
    ):
        print(f"5. Building undirected graph")
        self.num_class = num_class
        category_label, split, masks = self.get_category_label(split=split, shuffle=shuffle)
        self.user_nodes_feature = scale_feats(self.user_nodes_feature)
        self.product_nodes_feature = scale_feats(self.product_nodes_feature)
        feature = torch.cat((self.user_nodes_feature, self.product_nodes_feature), dim=0)
        src_nodes = []
        dst_nodes = []
        edge_feature = []
        for (i, j) in self.edges_dict.keys():
            src_nodes += [i, j]
            dst_nodes += [j, i]
            edge_feature += [self.edges_dict[(i, j)]['rating']] * 2
        graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
        graph.edata['rating'] = torch.tensor(edge_feature)
        graph.ndata['feat'] = feature
        graph.ndata['label'] = category_label
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = masks
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        # for test
        # self.count_users_price()
        # for test
        count_degrees(graph)
        return graph, (feature.shape[1], int(torch.max(category_label).item()) + 1)

    def build_graph_price_predict(
            self,
            num_class=5,
            split=(6, 2, 2),
            shuffle=True,
    ):
        print(f"5. Building undirected graph")
        price, price_label, split, masks = self.get_price_label(split=split, num_class=num_class, shuffle=shuffle)
        feature = scale_feats(torch.cat((self.user_nodes_feature, self.product_nodes_feature), dim=0))
        src_nodes = []
        dst_nodes = []
        edge_feature = []
        for (i, j) in self.edges_dict.keys():
            src_nodes += [i, j]
            dst_nodes += [j, i]
            edge_feature += [self.edges_dict[(i, j)]['rating']] * 2
        graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
        graph.edata['rating'] = torch.tensor(edge_feature)
        graph.ndata['feat'] = feature
        graph.ndata['label'] = price_label
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = masks
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        # normalize price for regression task
        used_price = price[masks[0] + masks[1] + masks[2]]
        mean_a = torch.mean(used_price)
        std_a = torch.std(used_price)
        n1 = (used_price - mean_a) / std_a
        price[masks[0] + masks[1] + masks[2]] = n1
        graph.ndata['price'] = price
        # for test
        # self.count_users_price()
        # for test
        count_degrees(graph)
        return graph, (feature.shape[1], int(torch.max(price_label).item()) + 1)

    def build_co_purchase_network_category_predict(
            self,
            num_class=5,
            split=(6, 2, 2),
            shuffle=True,
    ):
        print(f"5. Building undirected graph")
        self.num_class = num_class
        print([x['description'] for x in self.products_info_dict.values()])
        src_nodes = self.co_product_edges[:, 0]
        dst_nodes = self.co_product_edges[:, 1]
        graph = dgl.graph((src_nodes, dst_nodes))

        category_label, split, masks = self.get_category_label(split=split, shuffle=shuffle)
        feature = scale_feats(self.product_nodes_feature)

        graph.ndata['feat'] = feature
        graph.ndata['label'] = category_label
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = masks
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        # for test
        # self.count_users_price()
        # for test
        count_degrees(graph)
        # for i in range(graph.num_nodes()):
        #     print(graph.ndata['label'][i], end="")
        #     print(graph.ndata['label'][graph.successors(i)])
        return graph, (feature.shape[1], int(torch.max(category_label).item()) + 1)

    def build_co_purchase_network_price_predict(
            self,
            num_class=5,
            split=(6, 2, 2),
            shuffle=True,
    ):
        print(f"5. Building undirected graph")
        src_nodes = self.co_product_edges[:, 0]
        dst_nodes = self.co_product_edges[:, 1]
        graph = dgl.graph((src_nodes, dst_nodes))
        price, price_label, split, masks = self.get_price_label(split=split, num_class=num_class, shuffle=shuffle)
        feature = scale_feats(self.product_nodes_feature)
        graph.ndata['feat'] = feature
        graph.ndata['label'] = price_label
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = masks
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        # normalize price for regression task
        used_price = price[masks[0] + masks[1] + masks[2]]
        mean_a = torch.mean(used_price)
        std_a = torch.std(used_price)
        n1 = (used_price - mean_a) / std_a
        price[masks[0] + masks[1] + masks[2]] = n1
        graph.ndata['price'] = price
        # for test
        # self.count_users_price()
        # for test
        count_degrees(graph)
        return graph, (feature.shape[1], int(torch.max(price_label).item()) + 1)

    def save_preprocess_info(self, path=None):
        assert path is not None
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "id2users.npy"), self.id2users)
        np.save(os.path.join(path, "users_dict.npy"), self.users_dict)
        np.save(os.path.join(path, "id2products.npy"), self.id2products)
        np.save(os.path.join(path, "products_dict.npy"), self.products_dict)
        np.save(os.path.join(path, "edges_dict.npy"), self.edges_dict)
        np.save(os.path.join(path, "products_info_dict.npy"), self.products_info_dict)
        np.save(os.path.join(path, "id2catergories.npy"), self.id2catergories)
        np.save(os.path.join(path, "categories_dict.npy"), self.categories_dict)

    def load_preprocess_info(self, preprocess_path=None):
        self.id2users = np.load(os.path.join(preprocess_path, "id2users.npy"), allow_pickle=True).item()
        self.users_dict = np.load(os.path.join(preprocess_path, "users_dict.npy"), allow_pickle=True).item()
        self.id2products = np.load(os.path.join(preprocess_path, "id2products.npy"), allow_pickle=True).item()
        self.products_dict = np.load(os.path.join(preprocess_path, "products_dict.npy"), allow_pickle=True).item()
        self.edges_dict = np.load(os.path.join(preprocess_path, "edges_dict.npy"), allow_pickle=True).item()
        self.products_info_dict = np.load(os.path.join(preprocess_path, "products_info_dict.npy"),
                                          allow_pickle=True).item()
        self.id2catergories = np.load(os.path.join(preprocess_path, "id2catergories.npy"), allow_pickle=True).item()
        self.categories_dict = np.load(os.path.join(preprocess_path, "categories_dict.npy"), allow_pickle=True).item()
        self.num_users = len(self.id2users)
        self.num_products = len(self.id2products)
        if os.path.exists(os.path.join(preprocess_path, f"user_feature_{self.user_feature_type}.npy")):
            self.user_nodes_feature = torch.tensor(
                np.load(os.path.join(preprocess_path, f"user_feature_{self.user_feature_type}.npy"),
                        allow_pickle=True))
        if os.path.exists(os.path.join(preprocess_path, f"product_feature_{self.product_feature_type}.npy")):
            self.product_nodes_feature = torch.tensor(
                np.load(os.path.join(preprocess_path, f"product_feature_{self.product_feature_type}.npy"),
                        allow_pickle=True))
        if os.path.exists(os.path.join(preprocess_path, "co_product_edges.npy")):
            self.co_product_edges = torch.tensor(np.load(os.path.join(preprocess_path, "co_product_edges.npy"),
                                                         allow_pickle=True))

    def save_nodes_feature(self, save_path=None):
        assert save_path is not None
        if not os.path.exists(os.path.join(save_path,
                                           f"user_feature_{self.user_feature_type}.npy")) and self.user_nodes_feature is not None:
            print(f"   Saving user feature ({self.user_feature_type}) into {save_path}")
            np.save(os.path.join(save_path, f"user_feature_{self.user_feature_type}.npy"), self.user_nodes_feature)
        if not os.path.exists(os.path.join(save_path,
                                           f"product_feature_{self.product_feature_type}.npy")) and self.product_nodes_feature is not None:
            print(f"   Saving product feature ({self.product_feature_type}) into {save_path}")
            np.save(os.path.join(save_path, f"product_feature_{self.product_feature_type}.npy"),
                    self.product_nodes_feature)

    def filter_nodes_with_degree(self, filter_degree):
        src_nodes = []
        dst_nodes = []
        for (i, j) in self.edges_dict.keys():
            src_nodes += [i]
            dst_nodes += [j]
        graph = dgl.DGLGraph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
        degrees = (graph.in_degrees() + graph.out_degrees()).numpy()
        idx_reserve = np.where(degrees > filter_degree)[0]
        user_reserve = idx_reserve[idx_reserve < len(self.id2users)]
        product_reserve = idx_reserve[idx_reserve >= len(self.id2users)]
        self.num_users = len(user_reserve)
        self.num_products = len(product_reserve)
        return user_reserve, product_reserve

    def update_index(self, id2users, id2products, edges_dict, user_reserve, product_reserve):
        self.num_users = len(user_reserve)
        self.num_products = len(product_reserve)
        if len(user_reserve) == len(id2users.keys()) and len(product_reserve) == len(id2products.keys()):
            return
        if len(user_reserve) != len(id2users.keys()):
            self.users_dict = dict((id2users[old_id], new_id) for new_id, old_id in enumerate(user_reserve))
            self.id2users = dict((val, key) for key, val in self.users_dict.items())
        if len(product_reserve) != len(id2products.keys()):
            self.products_dict = dict(
                (id2products[old_id], new_id + self.num_users) for new_id, old_id in enumerate(product_reserve))
            self.id2products = dict((val, key) for key, val in self.products_dict.items())
        self.edges_dict = {}
        for (i, j), info in edges_dict.items():
            if i in user_reserve and j in product_reserve:
                self.edges_dict[(self.users_dict[id2users[i]], self.products_dict[id2products[j]])] = info

    def update_index_co_purchase(self, id2users, id2products, co_product_edges, user_reserve, product_reserve):
        self.num_users = len(user_reserve)
        assert self.num_users == 0
        self.num_products = len(product_reserve)
        self.users_dict = {}
        self.id2users = {}
        self.products_dict = dict((id2products[old_id], new_id) for new_id, old_id in enumerate(product_reserve))
        self.id2products = dict((val, key) for key, val in self.products_dict.items())
        self.co_product_edges = []
        for (j1, j2) in co_product_edges:
            self.co_product_edges.append(((self.products_dict[id2products[j1]], self.products_dict[id2products[j2]])))

    def get_user_nodes_feature(self, feature_dim, feature_type):
        if not hasattr(self, 'products_info_dict'):
            raise NotImplementedError('Please load meta data before get nodes feature!')
        if hasattr(self, 'feature_dim'):
            assert feature_dim == self.feature_dim
        else:
            self.feature_dim = feature_dim
        id2products = self.id2products
        edges_dict = self.edges_dict
        products_info_dict = self.products_info_dict
        print("\n--- Generating user nodes features ---")
        user_nodes_raw_data = {}
        for (i, j) in edges_dict.keys():
            info = products_info_dict[id2products[j]]
            if feature_type == "name+description":
                raw_data = info['name'] + info['description']
            else:
                raw_data = info[feature_type]
            if i not in user_nodes_raw_data.keys():
                user_nodes_raw_data[i] = [raw_data]
            else:
                user_nodes_raw_data[i].append(raw_data)
        user_nodes_text = []
        for i in range(len(user_nodes_raw_data)):
            user_nodes_text.append(" ".join(user_nodes_raw_data[i]))
        # print(user_nodes_text)
        self.user_nodes_feature = self.feat_tokenize(user_nodes_text)

    def get_product_nodes_feature(self, feature_dim, feature_type):
        if not hasattr(self, 'products_info_dict'):
            raise NotImplementedError('Please load meta data before get nodes feature!')
        if hasattr(self, 'feature_dim'):
            assert feature_dim == self.feature_dim
        else:
            self.feature_dim = feature_dim
        print("\n--- Generating product nodes features ---")
        id2products = self.id2products
        products_info_dict = self.products_info_dict
        product_nodes_text = []
        for j in id2products.keys():
            info = products_info_dict[id2products[j]]
            if feature_type == "name+description":
                raw_data = info['name'] + info['description']
            else:
                raw_data = info[feature_type]
            product_nodes_text.append(raw_data)
        self.product_nodes_feature = self.feat_tokenize(product_nodes_text)

    def merge_category(self):
        with open(f'{self.dataset_name}.txt', 'w') as f:  # 设置文件对象
            for x in self.categories_dict.keys():
                f.write(x + '\n')  # 将字符串写入文件中
        # raise NotImplementedError
        categories_dict = copy.deepcopy(self.categories_dict)
        merge_conditions = get_merge_category(self.num_class, cate_mode=1, dataset_name=self.dataset_name)
        not_merged_cates = []
        origin2new_cate_id = {}
        for cate, old_id in categories_dict.items():
            cate = cate.lower()
            flag_merged = False
            for new_id, key_words in enumerate(merge_conditions):
                for key_word in key_words:
                    if key_word in cate:
                        flag_merged = True
                        origin2new_cate_id[old_id] = new_id
                        self.categories_dict[cate] = new_id
                        break
                if flag_merged:
                    break
            if not flag_merged:
                not_merged_cates.append(cate)
        # print(not_merged_cates)
        print(f"Not merged cates: {len(not_merged_cates)}/{len(categories_dict.keys())}")
        # self.id2catergories = None  # id2categories should not be used since merge
        return origin2new_cate_id

    def get_category_label(self, split, shuffle=True):
        products_info_dict = self.products_info_dict
        id2products = self.id2products
        # merge category labels
        origin2new_cate_id = self.merge_category()
        # get category labels
        cate_labels = {}
        count_usable_label = 0
        count_valid_label = 0
        for j in id2products.keys():
            categories = products_info_dict[id2products[j]]['category']
            if categories is not None:
                count_usable_label += 1
                cate_labels[j] = []
                for old_id in categories:
                    if old_id in origin2new_cate_id.keys():
                        cate_labels[j].append(origin2new_cate_id[old_id])
                if len(cate_labels[j]) != 0:
                    count_valid_label += 1
                else:
                    # print(products_info_dict[id2products[j]]['description'])
                    # print([self.id2catergories[cate] for cate in categories])
                    pass
        print(f"usable label products: {count_usable_label}/{self.num_products}")
        print(f"valid label products: {count_valid_label}/{count_usable_label}")

        # choose only one label for each node
        cate_label = np.zeros(self.num_users + self.num_products) - 1  # default -1
        for j in id2products.keys():
            if j in cate_labels.keys():
                # choose smallest label as the j_th nodes label
                # cate_label[j] = sorted(cate_labels[j],reverse=True)[0 if len(cate_labels[j])==1 else 1]
                if len(set(cate_labels[j])) == 1:
                    cate_label[j] = cate_labels[j][0]
                else:
                    # do not want "appliances" cate
                    cleaned_cates = [x for x in cate_labels[j] if x != self.num_class - 1]
                    cate_label[j] = max(set(cleaned_cates), key=cleaned_cates.count)
                # cate_label[j] = max(set(cate_labels[j]), key=cate_labels[j].count)
        # set split idx
        idx = np.where(cate_label >= 0)[0]  # -1 for no label nodes
        if shuffle:
            random.seed(0)
            index = list(range(len(idx)))
            random.shuffle(index)
            idx = idx[index]
        total = len(idx)
        assert len(split) == 3
        train_idx = idx[:split[0] * total // np.sum(split)]
        val_idx = idx[split[0] * total // np.sum(split):(split[0] + split[1]) * total // np.sum(split)]
        test_idx = idx[(split[0] + split[1]) * total // np.sum(split):]
        print(f"Train samples: {len(train_idx)} Val samples: {len(val_idx)} Test samples: {len(test_idx)}")
        train_mask = [False] * cate_label.shape[0]
        val_mask = [False] * cate_label.shape[0]
        test_mask = [False] * cate_label.shape[0]
        for i in train_idx:
            train_mask[i] = True
        for i in val_idx:
            val_mask[i] = True
        for i in test_idx:
            test_mask[i] = True
        count_label(np.array(cate_label))
        return torch.tensor(cate_label, dtype=torch.long), (train_idx, val_idx, test_idx), (
            torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask))

    def get_price_label(self, split, num_class=10, shuffle=True):
        products_info_dict = self.products_info_dict
        id2products = self.id2products
        # zero for nodes have no price attribute
        price = np.zeros(self.num_users + self.num_products)
        for j in id2products.keys():
            info = products_info_dict[id2products[j]]
            price[j] = int(info['price']) if info['price'] is not None else 0
        # set split idx
        idx = np.where(price > 0)[0]
        if shuffle:
            random.seed(0)
            index = list(range(len(idx)))
            random.shuffle(index)
            idx = idx[index]
        total = len(idx)
        assert len(split) == 3
        train_idx = idx[:split[0] * total // np.sum(split)]
        val_idx = idx[split[0] * total // np.sum(split):(split[0] + split[1]) * total // np.sum(split)]
        test_idx = idx[(split[0] + split[1]) * total // np.sum(split):]
        print(f"Train samples: {len(train_idx)} Val samples: {len(val_idx)} Test samples: {len(test_idx)}")
        train_mask = [False] * price.shape[0]
        val_mask = [False] * price.shape[0]
        test_mask = [False] * price.shape[0]
        for i in train_idx:
            train_mask[i] = True
        for i in val_idx:
            val_mask[i] = True
        for i in test_idx:
            test_mask[i] = True
        # price into discrete labels
        price_bins = [0]
        sorted_price = np.sort(price[np.where(price > 0)])
        for i, p in enumerate(sorted_price):
            if i != 0 and i % (total // (num_class - 1)) == 0:
                price_bins.append(p)
        price_label = []
        for p in price:
            if p == 0:
                price_label.append(-1)
                continue
            ind = np.max(np.where(price_bins < p))
            price_label.append(ind)
        # count_price_label(np.array(price_label))
        # for i in set(list(price_label)):
        #     print(f"{i}:{list(price_label).count(i)}")
        # print(f"{list(price_label).count(-1)}/{len(price_label)}")
        # print(
        #     f"{list(np.array(price_label)[np.concatenate((train_idx, val_idx, test_idx))]).count(-1)}/{(np.concatenate((train_idx, val_idx, test_idx))).shape[0]}")
        return torch.tensor(price, dtype=torch.float), torch.tensor(price_label, dtype=torch.long), (
            train_idx, val_idx, test_idx), (
            torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask))

    def feat_tokenize(self, nodes_text):
        user_nodes_feature = []
        glove = GloVe(name='6B', dim=self.feature_dim)
        cnt_invalid_feature = 0
        for text in tqdm(nodes_text):
            text_ = text.replace("-", " ")
            words = nltk.word_tokenize(text_)
            if len(words) == 0:
                print(text)
            tensor = glove.get_vecs_by_tokens(words, lower_case_backup=True)
            non_zero_tensor = tensor[[not torch.all(tensor[i] == 0) for i in range(tensor.shape[0])], :]
            if non_zero_tensor.shape[0] == 0:
                # just for 调试, should be deleted later
                # if text_ == "Yonyko":
                #     text_ = "Play Game Controller"
                # elif text_ == "MAYPLUSS":
                #     text_ = "Organic bamboo fiber towel"
                # else:
                #     raise NotImplementedError(f"Error text: {text_}")
                # words = nltk.word_tokenize(text_)
                # tensor = glove.get_vecs_by_tokens(words, lower_case_backup=True)
                # just for 调试, should be deleted later
                cnt_invalid_feature += 1
                non_zero_tensor = torch.zeros((1, 300))
            user_nodes_feature.append(torch.mean(non_zero_tensor, dim=0))
        print(f"Total invalid nodes: {cnt_invalid_feature}/{len(nodes_text)}")
        return torch.stack(user_nodes_feature)

    def assure_no_duplicated_products(self, path):
        with open(path) as f:
            data = []
            for line in f.readlines():
                data.append(line)
        print(f"\nTotal repeated products: {len(data)}")
        repeated_products = {}
        for i, line in enumerate(data):
            for product in line[:-1].split(" "):
                repeated_products[product] = i
        self.repeated_products = repeated_products
        bin = []
        for product in self.products_dict.keys():
            if product in self.repeated_products.keys():
                new = self.repeated_products[product]
                for i in bin:
                    if new == i:
                        print(f"duplicated product: {product}")
                bin.append(self.repeated_products[product])
        print(len(bin))
        print(f'Duplicated products: {len(bin) - len(list(set(bin)))}')
        if len(bin) - len(list(set(bin))) != 0:
            raise NotImplementedError("Please process the duplicated products!")


Dataset_Name_Dict = {
    'ab': 'All_Beauty',
    'af': 'AMAZON_FASHION',
    'app': 'Appliances',
    'arts': 'Arts_Crafts_and_Sewing',
    'ggf': 'Grocery_and_Gourmet_Food',
    'lb': 'Luxury_Beauty',
    'mgz': 'Magazine_Subscriptions',
    'movies': 'Movies_and_TV',
    'sftw': 'Software',
    'sports': 'Sports_and_Outdoors',
    'toys': "Toys_and_Games"
}


def load_amazon_exp_dataset(dataset):
    mode = 'user_product_cate'
    user_feature_type = 'description'
    product_feature_type = 'name+description'
    name = dataset[7:]  # amazon-
    dataset_name = Dataset_Name_Dict[name]
    print(f"\n---Loading Amazon experimental dataset: {dataset_name}---")
    raw_data_root = '../SSL-GNN-Benchmark/data/Amazon_Product/' + dataset_name + '/'
    review_path = raw_data_root + dataset_name + '.json'
    meta_path = raw_data_root + 'meta_' + dataset_name + '.json'
    save_root = 'data/dataset/' + dataset_name
    duplicate_path = '../SSL-GNN-Benchmark/data/duplicates.txt'
    amazon_info = amazon_graph_info(
        name,
        filter_degree=5,
        feature_dim=300,
        save_path=save_root,
        preprocess_path=save_root,
        review_path=review_path,
        meta_path=meta_path,
        duplicate_path=None,
        graph_type=mode,
        user_feature_type=user_feature_type,
        product_feature_type=product_feature_type
    )
    if mode == 'user_product_cate':
        return amazon_info.build_graph_category_predict(
            split=[6, 2, 2],
            num_class=10,
            shuffle=True
        )
    elif mode == 'user_product_price':
        return amazon_info.build_graph_price_predict(
            split=[6, 2, 2],
            num_class=5,
            shuffle=True,
        )
    elif mode == 'co_product_cate':
        return amazon_info.build_co_purchase_network_category_predict(
            split=[6, 1, 2],
            num_class=10,
            shuffle=True,
        )
    elif mode == 'co_product_price':
        return amazon_info.build_co_purchase_network_price_predict(
            split=[6, 1, 2],
            num_class=5,
            shuffle=True,
        )
    else:
        raise NotImplementedError(f"Amazon graph building and training mode: {mode} not supported!")
