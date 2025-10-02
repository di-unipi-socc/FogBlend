% Order nodes by min HW (descending)
heuristicNodes(OrderedNodes) :-
	findall((MinHW, N),
		( node(N, _, HWCaps, _),
		  HWCaps = (CPU, GPU, Storage),
		  MinHW is min(CPU, min(GPU, Storage))
		),
		NodeHWPairs),
	sort(1, @>=, NodeHWPairs, SortedPairs),                      
	findall(N, member((_, N), SortedPairs), OrderedNodes).


% Order services by their bw requirements (descending)
heuristicServices(Services, OrderedServices) :-
	findall((ReqBW, N),
		( member(N, Services),
		  findall(BW, s2s(N, _, _, BW), BWsOut),  
		  findall(BW, s2s(_, N, _, BW), BWsIn),     
		  sum_list(BWsOut, ReqBWOut),
		  sum_list(BWsIn, ReqBWIn),
		  ReqBW is ReqBWIn + ReqBWOut
		),
		ServiceBWPairs),
	sort(1, @>=, ServiceBWPairs, SortedPairs),                      
	findall(N, member((_, N), SortedPairs), OrderedServices).