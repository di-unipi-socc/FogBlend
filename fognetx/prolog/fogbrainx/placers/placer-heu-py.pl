placement(Services, P, Alloc, NewPlacement) :-
	once(preprocessing(Services, OrderedNodes, OrderedServices)), 
	hplacement(OrderedServices, P, OrderedNodes, Alloc, NewPlacement).

hplacement([S|Ss], P, OrderedNodes, (AllocHW,AllocBW), Placement) :-
	member(N, OrderedNodes),
	nodeOk(S,N,P,AllocHW), linksOk(S,N,P,AllocBW), 
	hplacement(Ss, [on(S,N)|P], OrderedNodes, (AllocHW,AllocBW), Placement).
hplacement([],P,_,_,P).

% Preprocessing to get ordered nodes and services
preprocessing(Services, OrderedNodes, OrderedServices) :-
	orderNodesHW(OrderedNodes),
	orderServicesBW(Services, OrderedServices).


% Order nodes by min HW (descending)
orderNodesHW(OrderedNodes) :-
	findall((MinHW, ID),
		( node(ID, _, HWCaps, _),
		  HWCaps = (CPU, RAM, Storage),
		  MinHW is min(CPU, min(RAM, Storage))
		),
		NodeHWPairs),
	sort(1, @>=, NodeHWPairs, SortedPairs),                      
	findall(ID, member((_, ID), SortedPairs), OrderedNodes).


% Order nodes by total outgoing bandwidth (descending)
orderNodesBW(OrderedNodes) :-
    findall((TotalBW, ID),
        ( node(ID, _, _, _),
          findall(BW, link(ID, _, _, BW), BWs),        
          sum_list(BWs, TotalBW)
        ),
        NodeBWPairs),
    sort(1, @>=, NodeBWPairs, SortedPairs),                      
    findall(ID, member((_, ID), SortedPairs), OrderedNodes).

% Order services by their bw requirements (descending)
orderServicesBW(Services, OrderedServices) :-
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