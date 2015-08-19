from collections import defaultdict




keypoints = [1,2,3,4,5,6]
n = {}
n["1"] = set(["1", "2", "3"])
n["2"] = set(["2", "1", "4"])
n["3"] = set(["3", "1", "4"])
n["4"] = set(["4", "2", "3"])

n["6"] = set(["6"])


def unify(neighbourMap):
	for key in neighbourMap.iterkeys(): 
			b = neighbourMap[key]
			for values in neighbourMap.itervalues(): 	
				if len(b.intersection(values)) > 0:
					b = b.union(values)
					neighbourMap[key] = b
			
	res = []
	for values in neighbourMap.itervalues(): 	
		if  values not in res:
			res.append(values)
	return res
	
	
print unify(n)