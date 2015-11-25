import urbs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def figure(prob, com, sit, timesteps=None, power_unit='MW', energy_unit='MWh'):
    """Plot a stacked timeseries of commodity balance and storage.

    Creates a stackplot of the energy balance of a given commodity, together
    with stored energy in a second subplot.

    Args:
        prob: urbs model instance
        com: commodity name to plot
        sit: site name to plot
        timesteps: optional list of  timesteps to plot; default: prob.tm
        power_unit: optional string for unit; default: 'MW'
        energy_unit: optional string for storage plot; default: 'MWh'

    Returns:
        fig: figure handle
    """

    if timesteps is None:
        # default to all simulated timesteps
        timesteps = sorted(urbs.get_entity(prob, 'tm').index)

    # FIGURE
    fig = plt.figure(figsize=(16, 8))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    created, consumed, stored, imported, exported = urbs.get_timeseries(
        prob, com, sit, timesteps)

    costs, cpro, ctra, csto = urbs.get_constants(prob)

    # move retrieved/stored storage timeseries to created/consumed and
    # rename storage columns back to 'storage' for color mapping
    created = created.join(stored['Retrieved'])
    consumed = consumed.join(stored['Stored'])
    created.rename(columns={'Retrieved': 'Storage'}, inplace=True)
    consumed.rename(columns={'Stored': 'Storage'}, inplace=True)

    # only keep storage content in storage timeseries
    stored = stored['Level']

    # add imported/exported timeseries
    created = created.join(imported)
    consumed = consumed.join(exported)

    # move demand to its own plot
    demand = consumed.pop('Demand')

    # remove all columns from created which are all-zeros in both created and
    # consumed (except the last one, to prevent a completely empty frame)
    for col in created.columns:
        if not created[col].any() and len(created.columns) > 1:
            if col not in consumed.columns or not consumed[col].any():
                created.pop(col)

    # sorting plot elements
    created, consumed = sort_plot_elements(created, consumed)

    # PLOT CREATED
    ax0 = plt.subplot(gs[0])
    sp0 = ax0.stackplot(created.index, created.as_matrix().T, linewidth=0.15)

    # Unfortunately, stackplot does not support multi-colored legends itself.
    # Therefore, a so-called proxy artist - invisible objects that have the
    # correct color for the legend entry - must be created. Here, Rectangle
    # objects of size (0,0) are used. The technique is explained at
    # http://stackoverflow.com/a/22984060/2375855
    proxy_artists = []
    for k, commodity in enumerate(created.columns):
        commodity_color = urbs.to_color(commodity)

        sp0[k].set_facecolor(commodity_color)
        sp0[k].set_edgecolor(urbs.to_color('Decoration'))

        proxy_artists.append(mpl.patches.Rectangle(
            (0, 0), 0, 0, facecolor=commodity_color))

    # label
    ax0.set_title('Energy balance of {} in {}'.format(com, sit))
    ax0.set_ylabel('Power ({})'.format(power_unit))

    # legend
    # add "only" consumed commodities to the legend
    lg_items = tuple(created.columns)
    for item in consumed.columns:
        # if item not in created add to legend, except items
        # from consumed which are all-zeros
        if item in created.columns or not consumed[item].any():
            pass
        else:
            # add item/commodity is not consumed
            commodity_color = urbs.to_color(item)
            proxy_artists.append(mpl.patches.Rectangle(
                (0, 0), 0, 0, facecolor=commodity_color))
            lg_items = lg_items + (item,)

    lg = ax0.legend(proxy_artists,
                    lg_items,
                    frameon=False,
                    ncol=len(proxy_artists),
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.01))
    plt.setp(lg.get_patches(), edgecolor=urbs.to_color('Decoration'),
             linewidth=0.15)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # PLOT CONSUMED
    sp00 = ax0.stackplot(consumed.index, -consumed.as_matrix().T,
                         linewidth=0.15)

    # color
    for k, commodity in enumerate(consumed.columns):
        commodity_color = urbs.to_color(commodity)

        sp00[k].set_facecolor(commodity_color)
        sp00[k].set_edgecolor((.5, .5, .5))

    # PLOT DEMAND
    ax0.plot(demand.index, demand.values, linewidth=1.2,
             color=urbs.to_color('Demand'))

    # SECONDARY PLOT
    #ax1, sp1 = plot_storage(gs, ax0, stored, energy_unit, csto, sit, com)
    ax1, sp1 =  plot_variable_costs(gs, ax0, prob, sit, created)
    #ax1, sp1 =  plot_costs(gs, ax0, prob, sit)

    # axis properties
    axis_properties(timesteps, ax0, ax1)

    return fig


def sort_plot_elements(created, consumed):
    """Sort plot elements ascending with variance

    Sorts created and consumed ascending with variance.
    It places base load at the bottom and peak load at the top.
    This enhances clearity of the plots.

    Args:
        created
        consumed

    Returns:
        created (sorted)
        consumed (sorted)
    """

    # sorting created
    if len(created.columns) > 1:
        # calculate standard deviation
        std = pd.DataFrame(np.zeros_like(created.tail(1)),
                           index=created.index[-1:]+1, columns=created.columns)
        for col in std.columns:
            std[col] = np.std(created[col])
        # sort created ascencing with std i.e. base load first
        created = created.append(std)
        new_columns = created.columns[created.ix[created.last_valid_index()].argsort()]
        created = created[new_columns][:-1]

    # sorting consumed
    if len(consumed.columns) > 1:
        # calculate standard deviation
        std = pd.DataFrame(np.zeros_like(consumed.tail(1)),
                           index=consumed.index[-1:]+1, columns=consumed.columns)
        for col in std.columns:
            std[col] = np.std(consumed[col])
        # sort consumed ascencing with std i.e. base load first
        consumed = consumed.append(std)
        new_columns = consumed.columns[consumed.ix[consumed.last_valid_index()].argsort()]
        consumed = consumed[new_columns][:-1]

    return created, consumed


def plot_storage(gs, ax0, stored, energy_unit, csto, sit, com):
    """Plot storage in secondary plot

    Plots the storage timeseries in a subplot.

    Args:
        gs, ax0, stored, energy_unit, csto, sit, com

    Returns:
        ax1, sp1
    """

    ax1 = plt.subplot(gs[1], sharex=ax0)
    sp1 = ax1.stackplot(stored.index, stored.values, linewidth=0.15)

    # color
    sp1[0].set_facecolor(urbs.to_color('Storage'))
    sp1[0].set_edgecolor(urbs.to_color('Decoration'))

    # labels & y-limits
    ax1.set_xlabel('Time in year (h)')
    ax1.set_ylabel('Energy ({})'.format(energy_unit))
    try:
        ax1.set_ylim((0, 0.5 + csto.loc[sit, :, com]['C Total'].sum()))
    except KeyError:
        pass

    return ax1, sp1


def plot_variable_costs(gs, ax0, prob, sit, created):
    """Plot variable costs in secondary plot

    Plots the variable costs in a subplot.

    Args:
        gs, ax0, stored, energy_unit, csto, sit, com

    Returns:
        ax1, sp1
    """

    ax1 = plt.subplot(gs[1], sharex=ax0)

    buy_tuples = urbs.commodity_subset(prob.com_tuples, prob.com_buy)
    buy_prices = urbs.get_com_price(prob, buy_tuples)
    buy_col = buy_prices.columns.isin([(sit,'Elec buy','Buy')])
    buy_color = urbs.to_color('Purchase')
	
    if buy_col.any() == False:
        sp1 = ax1.plot([],[])
    else:
        sp1 = ax1.plot(buy_prices.index, buy_prices.loc[:,buy_col].values, linewidth=2, color=buy_color)

    sell_tuples = urbs.commodity_subset(prob.com_tuples, prob.com_sell)
    sell_prices = urbs.get_com_price(prob, sell_tuples)
    sell_col = sell_prices.columns.isin([(sit,'Elec sell','Sell')])
    sell_color = urbs.to_color('Feed-In')

    if sell_col.any() == False:
        sp1 = ax1.plot([],[])
    else:
        sp1 = ax1.plot(sell_prices.index, sell_prices.loc[:,sell_col].values, linewidth=2, color=sell_color)

    #for k, process in enumerate(created.columns):
        # get process-tuple -> get var-cost
        # get process-commodity-tuple -> get elec-out-ratio -> get commodity
        # get commodity-tuple with commodity -> get price
        # calculate total-var-cost = (var-cost + price) / elec-out-ratio
        # plot total-var-cost
	
    #for k, process in enumerate(consumed.columns):
        # same procedure

    # how to deal with imported/exported energy?

    # maybe also plotting variable costs of stored energy usage
    # 1) calculate costs of storage (losses and var-cost)
    # 2) plot total-var-cost + sto-var-cost with original color
    # 3) plot total-var-cost + sto-var-cost dashed with storage color       
	
    # labels & y-limits
    ax1.set_xlabel('Time in year (h)')
    ax1.set_ylabel('Costs (Euro/MWh)')

    return ax1, sp1


def plot_costs(gs, ax0, prob, sit):
    """Plot cumulative costs in secondary plot

    Plots the cumulative costs in a subplot.

    Args:
        gs, ax0, stored, energy_unit, csto, sit, com

    Returns:
        ax1, sp1
    """
    
    ax1 = plt.subplot(gs[1], sharex=ax0)
    
    # plot fix costs
    # 1) get process-tuple -> get fix-cost -> get inv-cost & wacc & depreciation
    # 2) calulate total-fix-cost with investment costs weighed with m.tm
    
    # plot cummulative variable costs on top
    # 3) calculate total fix cost
    # 4) get total-var-cost like in plot_variable_cost()
    # 5) get quantities
    # 6) np.cumsum(quantity * total-var-buy_price) + total-fix-cost
    
    sp1 = ax1.plot([],[])
    
	# labels & y-limits
    ax1.set_xlabel('Time in year (h)')
    ax1.set_ylabel('Costs (Euro)')    
    
    return ax1, sp1


def axis_properties(timesteps, ax0, ax1):
    """Customize axis properties

    Customize the axis properties (limits and ticks)

    Args:
        timesteps, ax0, ax1

    Returns:
        ...
    """

    # make xtick distance duration-dependent
    if len(timesteps) > 26*168:
        steps_between_ticks = 168*4
    elif len(timesteps) > 3*168:
        steps_between_ticks = 168
    elif len(timesteps) > 2 * 24:
        steps_between_ticks = 24
    elif len(timesteps) > 24:
        steps_between_ticks = 6
    else:
        steps_between_ticks = 3
    xticks = timesteps[::steps_between_ticks]

    # set limits and ticks for both axes
    for ax in [ax0, ax1]:
        # ax.set_axis_bgcolor((0, 0, 0, 0))
        plt.setp(ax.spines.values(), color=urbs.to_color('Decoration'))
        ax.set_frame_on(False)
        ax.set_xlim((timesteps[0], timesteps[-1]))
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, 'major', color=urbs.to_color('Grid'),
                      linestyle='-')
        ax.yaxis.grid(True, 'major', color=urbs.to_color('Grid'),
                      linestyle='-')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # group 1,000,000 with commas
        group_thousands = mpl.ticker.FuncFormatter(
            lambda x, pos: '{:0,d}'.format(int(x)))
        ax.yaxis.set_major_formatter(group_thousands)
