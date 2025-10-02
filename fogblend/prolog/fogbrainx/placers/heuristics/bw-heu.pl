% Order nodes by total outgoing bandwidth (descending)
heuristicNodes(OrderedNodes) :-
    findall((TotalBW, N),
        ( node(N, _, _, _),
          findall(BW, link(N, _, _, BW), BWs),        
          sum_list(BWs, TotalBW)
        ),
        NodeBWPairs),
    sort(1, @>=, NodeBWPairs, SortedPairs),                      
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