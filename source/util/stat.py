import numpy as np

def sample_from_dist(num_by_id, sample_num = 1):
    # bisect search
    def bisect_search(sorted_na, value):
        '''
        Do bisect search
        :param sorted_na: cumulated sum array
        :param value: random value
        :return: the index that sorted_na[index-1]<=value<sorted_na[index] with defining sorted_na[-1] = -1
        '''
        if len(sorted_na) == 1:
            return 0
        left_index = 0
        right_index = len(sorted_na)-1

        while right_index-left_index > 1:
            mid_index = (left_index + right_index) / 2
            # in right part
            if value > sorted_na[mid_index]:
                left_index = mid_index
            elif value < sorted_na[mid_index]:
                right_index = mid_index
            else:
                return min(mid_index+1,right_index)
        return right_index
    id, num = zip(*num_by_id)
    num = np.asarray(num, dtype = "int64")
    cum_num = num.cumsum()
    rvs = np.random.randint(0, cum_num[-1],size = sample_num)
    sample_ids = []
    for rv in rvs:
        if len(id) < 20000: # This value is obtained by test
            index = np.argmin(np.abs(cum_num-rv))
            if rv >= cum_num[index]:
                index += 1
        else:
            index = bisect_search(cum_num, rv)
        sample_ids.append(id[index])
    return sample_ids

