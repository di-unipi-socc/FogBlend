% Order nodes by min HW (descending)
heuristicNodes(OrderedNodes) :-
	findall((MinHW, ID),
		( node(ID, _, HWCaps, _),
		  HWCaps = (CPU, RAM, Storage),
		  MinHW is min(CPU, min(RAM, Storage))
		),
		NodeHWPairs),
	sort(1, @>=, NodeHWPairs, SortedPairs),                      
	findall(ID, member((_, ID), SortedPairs), OrderedNodes).


% Order services by their bw requirements (descending)
heuristicServices(Services, OrderedServices) :-
	findall((ReqBW, ID),
		( member(ID, Services),
		  findall(BW, s2s(ID, _, _, BW), BWsOut),  
		  findall(BW, s2s(_, ID, _, BW), BWsIn),     
		  sum_list(BWsOut, ReqBWOut),
		  sum_list(BWsIn, ReqBWIn),
		  ReqBW is ReqBWIn + ReqBWOut
		),
		ServiceBWPairs),
	sort(1, @>=, ServiceBWPairs, SortedPairs),                      
	findall(ID, member((_, ID), SortedPairs), OrderedServices).