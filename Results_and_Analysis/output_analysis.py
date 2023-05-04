import os
import time
import json
import pickle
import numpy as np
from sklearn import svm
from scipy import stats
from scipy.spatial import distance
import copy
import csv
import time

os.chdir("C:/Users/Damian/Documents/School/Winter 2023 - Grad School/IFT 6164 - Adversarial Learning/Project/Repo/Results_and_Analysis")
dist_type = "cosine"
ref_type = "avg"
img_names_lst = []

def load_data(dp_prefix):
    w_lst = []
    w_labels = []
    img_names_lst.clear()
    for c in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]: # "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        datapath = dp_prefix+c
        folder_name_lst = os.listdir(datapath)
        folder_name_lst.sort(key=int)
        img_names_lst.extend(folder_name_lst)
        for fldr_name in folder_name_lst:
            w_lst.append(np.load(datapath+"/"+fldr_name+"/projected_w.npz")['w'][0][0])
            w_labels.append(c)
    return w_lst, w_labels

def get_stats(w_lst):
    """
    returns avg vec, mean dist, std for all elements across all class
    """
    np_wlst = np.array(w_lst)
    n, m = np_wlst.shape
    avg_vec = np.average(np_wlst, axis=0)
    median_vec = np.median(np_wlst, axis=0)
    statspack = (avg_vec, 0, 0, median_vec)
    dist = dist_from_ref(statspack, w_lst)
    mean_dist = np.mean(dist)
    std = np.std(dist)
    return (avg_vec, mean_dist, std)

def compare_loop(comp):
    std_lst = []
    end_bound = 3
    step = .5
    std_bounds = [0,.5, 1, 1.5, 2, 2.5, end_bound]
    for i in std_bounds[:-1]:
        std_lst.append(sum([abs(x)<i+step and abs(x)>=i for x in comp]))   
    std_lst.append(sum([abs(x)>=end_bound for x in comp])) # greater than 3 std
    return std_lst

def dist_from_ref(statspack, lst):
    refvec = statspack[0] if ref_type == "avg" else statspack[3] #median
    if dist_type=="cosine":
        dist = [distance.cosine(i, refvec) for i in lst]
    elif dist_type == "l2":
        dist = [distance.euclidean(i, refvec) for i in lst]
    elif dist_type == "wasserstein":
        dist = [stats.wasserstein_distance(i, refvec) for i in lst]
    return dist

def compare(vec_lst, statspack):
    """
    returns the count of vectors in each standard deviation from class average vector

    uses cosine distance
    """
    mean_dist = statspack[1]
    std = statspack[2]

    dist = dist_from_ref(statspack, vec_lst)
    comp = abs((dist-mean_dist)/std)
    std_lst = compare_loop(comp)
    
    return std_lst

def get_stats_per_class(w_lst):
    """
    returns avg vec, mean dist, std, and mean vec for each class
    """
    np_wlst = np.array(w_lst)
    sub_wlst = np.split(np_wlst, 10)
    per_class_stats_lst = []
    for c in sub_wlst:
        avg_vec = np.average(c, axis=0)
        median_vec = np.median(w_lst, axis=0)
        statspack = (avg_vec, 0, 0, median_vec)
        dist = dist_from_ref(statspack, c)
        mean_dist = np.mean(dist)
        std = np.std(dist)
        per_class_stats_lst.append((avg_vec, mean_dist, std, median_vec))
    return per_class_stats_lst

def compare_per_class(vec_lst, statspack):
    
    np_veclst = np.array(vec_lst)
    sub_veclst = np.split(np_veclst, 10)
    per_class_std = [] 
    for i in range(len(sub_veclst)):
        std_lst = []
        avg_vec, mean_dist, std, median_vec = statspack[i]
        dist = dist_from_ref(statspack[i], sub_veclst[i])
        
        comp = abs((dist-mean_dist)/std)
        std_lst = compare_loop(comp)
        per_class_std.append(std_lst)
        with open('std_count_dump_false.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(per_class_std)

    return per_class_std
    
def closest_clean_vec(clean_lst, adv_lst, clean_labels, type):
    """returns a list, in order of adv_lst, of tuples: (distance, closest clean vec label)"""
    closest_vec_lst = []
    for adv in adv_lst:
        min = float("inf")
        label = -1
        for i in range(len(clean_lst)):
            if type=="cosine":
                dist = distance.cosine(adv, clean_lst[i])
            elif type == "l2":
                dist = distance.euclidean(adv, clean_lst[i])
            elif type == "wasserstein":
                dist = stats.wasserstein_distance(adv, clean_lst[i])
            if dist < min:
                min = dist
                label = clean_labels[i]
        closest_vec_lst.append((min, label))

    return closest_vec_lst

def svm_func(w_lst, w_labels, adv_lst, adv_labels):
    """returns bool array indicating which predictions are correct or wrong"""
    clf = svm.SVC()
    clf.fit(w_lst, w_labels)
    result = clf.predict(adv_lst)
    return ([r==l for r, l in zip(result, adv_labels)], clf)

def closest_vec(closest_clean_vec, adv_labels):
    """returns a list of # of adv. vecs. whose closest clean vec is in the correct class.  Last element of the list is the total number of correct adv. vec."""
    advec_in_class_count_total = sum([a == c[1] for a, c in zip(adv_labels, closest_clean_vec)])
    sub_classlst = np.split(closest_clean_vec, 10)
    sub_advlst = np.split(np.array(adv_labels), 10)
    advec_in_class_count_perclass = []
    for sub_a, sub_c in zip(sub_advlst, sub_classlst):
        total_advec_in_class_count = sum([a == c[1] for a, c in zip(sub_a, sub_c)])
        advec_in_class_count_perclass.append(total_advec_in_class_count)
    advec_in_class = advec_in_class_count_perclass
    advec_in_class.append(advec_in_class_count_total)
    return advec_in_class

def get_predicted_classes(attack_type):
    attack_dict = {"subs_fgsm_01":"subs_fgsm_01_adv_preds",
                   "sces":"sces_ensemble_02_adv_preds",
                   "spes":"spes_ensemble_02_adv_preds",
                   "fgsm_01_cnn": "cnn_fgsm_01_adv_preds",
                   "fgsm_01_resnet":"attacks_resnet_eps_01_adv_preds",
                   "fgsm_005_resnet": "attacks_resnet_eps_005_adv_preds"}
    f = open(attack_dict[attack_type]+".json")
    json_dict = json.load(f)
    pred_lst = [v for k,v in json_dict.items()]
    return pred_lst

def dist_of_predicted_class(adv_lst, per_class_stats_lst, dist_from_avg, attack_type):
    pred_lst = np.array(get_predicted_classes(attack_type))
    comp_lst = []
    for i in range(10):
        scan_lst = copy.deepcopy(pred_lst)
        scan_lst[i*50:(i+1)*50] = -1
        pred_tuple =  [(adv_lst[index], index) for (index, item) in enumerate(scan_lst) if item == i]
        avg_vec, mean_dist, std, median_vec = per_class_stats_lst[i]
        dist = dist_from_ref(per_class_stats_lst[i], [t[0] for t in pred_tuple])
        comp = abs((dist-mean_dist)/std)
        for i,pt in enumerate([t[1] for t in pred_tuple]):
            dist_from_avg[pt] = dist[i]
        comp_lst.append(comp)
    return comp_lst

def car_to_horse(w_lst):
    sub_lst = np.split(np.array(w_lst), 10)
    car_lst = sub_lst[1]
    horse_lst = sub_lst[7]
    avg_horse = np.average(horse_lst, axis=0)
    statpack = (avg_horse, 0, 0, 0)
    dist = dist_from_ref(statpack, car_lst)
    min_index = np.argmin(dist)
    car_vec = car_lst[min_index]
    step = (avg_horse - car_vec)/100
    with open("results/svm_model.pickle", 'rb') as fid:
        svm_model = pickle.load(fid)
    for i in range(100):
        result = svm_model.predict([car_vec])
        if result[0] != '1':
            print(i)
            break
        car_vec += step
    np.save("temp/car_to_horse_w.npy", car_vec)

def strict_stats(w_lst):
    stats_lst = get_stats_per_class(w_lst)
    sub_lst = np.split(np.array(w_lst), 10)
    strict_avg_lst = []
    for statspack, vec_lst in zip(stats_lst, sub_lst):
        dist = dist_from_ref(statspack, vec_lst)
        comp = (dist-statspack[1])/statspack[2]
        weeded_lst = [vec for i, vec in enumerate(vec_lst) if comp[i] <=1.0]    
        s_avg = np.average(weeded_lst, axis=0)
        temp_statspack = (s_avg, 0, 0, 0)
        new_dist = dist_from_ref(statspack, vec_lst)
        s_mean = np.mean(new_dist)
        s_std = np.std(new_dist)
        strict_avg_lst.append((s_avg,s_mean,s_std,0))
    return strict_avg_lst

def dist_from_strict_avg(strict_stats_lst, w_lst):
    sub_lst = np.split(np.array(w_lst), 10)
    dist_lst = []
    for ssl, vec_lst in zip(strict_stats_lst, sub_lst):
        dist_lst.append(compare(vec_lst, ssl))
    with open('std_count_dump_true.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(dist_lst)
    return dist_lst

def std_func(vec_lst, statspack):
    np_veclst = np.array(vec_lst)
    sub_veclst = np.split(np_veclst, 10)
    per_class_std = [] 
    for i in range(len(sub_veclst)):
        std_lst = []
        avg_vec, mean_dist, std, median_vec = statspack[i]
        dist = dist_from_ref(statspack[i], sub_veclst[i])
        std_from = abs((dist-mean_dist)/std)
        per_class_std.append(dist)

    return per_class_std

def adv_img_info_csv(clean_lst, adv_lst, sts_per_class, attack_type):
    dist_from_avg_pred = [None]*500
    strict_dist_from_avg_pred = [None]*500
    pred_lst = get_predicted_classes(attack_type)
    label = -1
    f = open("adv_img_info_"+attack_type+".csv", "w")
    f.write("src,name,pred,dist_from_source,dist_from_pred,strict_dist_from_source,strict_dist_from_avg_pred\n")
    strict_pack = strict_stats(clean_lst)
    std_comp_per_class_adv = np.array(std_func(adv_lst, sts_per_class)).flatten()
    strict_std_comp_per_class_adv = np.array(std_func(adv_lst, strict_pack)).flatten()
    dist_of_predicted_class(adv_lst, sts_per_class, dist_from_avg_pred, attack_type)
    dist_of_predicted_class(adv_lst, strict_pack, strict_dist_from_avg_pred, attack_type)
    for i, t in enumerate(zip(img_names_lst, pred_lst, std_comp_per_class_adv,dist_from_avg_pred,strict_std_comp_per_class_adv, strict_dist_from_avg_pred)):
        if i % 50 == 0:
            label +=1
        std_comp_per_class_adv[i]
        f.write("{0},{1},{2},{3},{4},{5},{6}\n".format(label,t[0],t[1],t[2],t[3],t[4],t[5]))
    f.close()

def save_status(dist_type, ref_type, attack_type, strict):
    status_dict = {"dist_type":dist_type, "ref_type": ref_type, "attack_type":attack_type, "strict":strict}
    with open("status.json", "w") as outfile:
        json.dump(status_dict, outfile)

if __name__ == "__main__":
    start = time.time()
    dist_type = "wasserstein" #cosine, l2, wasserstein
    ref_type = "avg" #avg, median
    attack_type = "subs_fgsm_01" # subs_fgsm_01, sces, spes, fgsm_01_cnn, fgsm_01_resnet, fgsm_005_resnet
    strict = False
    save_status(dist_type, ref_type, attack_type, strict)

    print("start: ", time.time() - start)
    train_lst, train_labels = load_data("out_train/")
    clean_lst, clean_labels = load_data("out_test/")
    print("after load: ", time.time() - start)
    adv_lst, adv_labels = load_data("out_"+attack_type+"/") ######### CHECK YOU ARE USING THE RIGHT DATA !!!!! #######################
    #car_to_horse(clean_lst)

    if strict:
        strict_dist_from_avg_pred = [None]*500
        strict_stats_pack = strict_stats(clean_lst)
        dist_from_strict_avg_lst_clean = dist_from_strict_avg(strict_stats_pack, clean_lst)
        dist_from_strict_avg_lst_adv = dist_from_strict_avg(strict_stats_pack, adv_lst)
        np.save("results/"+attack_type+"_std_strict_comp_per_class_clean_"+dist_type+"_"+ref_type+".npy", dist_from_strict_avg_lst_clean)
        np.save("results/"+attack_type+"_std_strict_comp_per_class_adv_"+dist_type+"_"+ref_type+".npy", dist_from_strict_avg_lst_adv)
        pred_class_dist = dist_of_predicted_class(adv_lst, strict_stats_pack, strict_dist_from_avg_pred, attack_type)
        np.save("results/"+attack_type+"_strict_dist_of_predicted_class_per_class_"+dist_type+"_"+ref_type+".npy", pred_class_dist)

        comp_lst = []
        for i in range(10):
            comp = compare_loop(pred_class_dist[i])
            comp_lst.append(comp)
        with open('std_count_dump_true.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(comp_lst)
        
    else:
        bool_predict, svm_model = svm_func(train_lst, train_labels, adv_lst, adv_labels)
        np.save("results/svm_results_trainlst.npy", bool_predict)
        pickle.dump(svm_model, open("results/"+attack_type+"_svm_model.pickle", "wb"))

        #sts  = get_stats(clean_lst)
        #np.save("results/stats_"+dist_type+".npy", sts)
        #std_lst = compare(adv_lst, sts)

        sts_per_class = get_stats_per_class(train_lst)
        #avg_vec_lst = [stats_tuple[0] for stats_tuple in sts_per_class]
        #np.save("results/avg_vec_"+dist_type+"_"+ref_type+"_"+attack_type+".npy", avg_vec_lst)
        adv_img_info_csv(clean_lst, adv_lst, sts_per_class, attack_type)

        dist_from_avg_pred = [None]*500
        pred_class_dist = dist_of_predicted_class(adv_lst, sts_per_class, dist_from_avg_pred, attack_type)
        np.save("results/"+attack_type+"_dist_of_predicted_class_per_class_"+dist_type+"_"+ref_type+".npy", pred_class_dist)
        

        std_comp_per_class_clean = compare_per_class(clean_lst, sts_per_class)
        std_comp_per_class_adv = compare_per_class(adv_lst, sts_per_class)
        np.save("results/"+attack_type+"_std_comp_per_class_clean_"+dist_type+"_"+ref_type+".npy", std_comp_per_class_clean)
        np.save("results/"+attack_type+"_std_comp_per_class_adv_"+dist_type+"_"+ref_type+".npy", std_comp_per_class_adv)

        #ccv = closest_clean_vec(clean_lst, adv_lst, clean_labels, dist_type)
        #np.save("results/closest_clean_vec_"+dist_type+".npy", ccv)
        #closest_clean_vec = np.load("results/closest_clean_vec_"+dist_type+".npy")
#   
        #ccv = np.array(ccv)
        #advec_in_class = closest_vec(ccv, adv_labels)
        #np.save("results/advec_in_class_"+dist_type+".npy", advec_in_class)

    print(attack_type+" "+str(strict)+" fin")
    #np.save("results/compare.npy", comp)
    