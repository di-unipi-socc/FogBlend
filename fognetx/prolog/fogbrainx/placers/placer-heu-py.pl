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
	heuristicNodes(OrderedNodes),
	heuristicServices(Services, OrderedServices).