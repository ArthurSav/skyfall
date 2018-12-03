import React, {Component} from 'react'
import {FlatList, Image, StyleSheet, Text, ToolbarAndroid, View} from 'react-native'

export default class Generated extends Component {

    static navigationOptions = {header: null}

    renderItem(data) {
        let {item, index} = data;
        return (
            <View style={styles.itemBlock}>
                <Image source={{uri: "https://avatars0.githubusercontent.com/u/583598?s=460&v=4"}} style={styles.itemImage}/>
                <View style={styles.itemMeta}>
                    <Text style={styles.itemName}>{item.key}</Text>
                    <Text style={styles.itemLastMessage}>This is comment {index}</Text>
                </View>
            </View>
        )
    }

    render() {
        return (
            <View style={{marginTop: 24}}>
                % for component in components:
                    ${component}
                % endfor
            </View>
        )
    }
}

const styles = StyleSheet.create({
    toolbar: {
        backgroundColor: '#2196F3',
        height: 56
    },
    container: {
        flex: 1,
        marginTop: 20,
    },
    itemBlock: {
        flexDirection: 'row',
        paddingBottom: 5,
    },
    itemImage: {
        width: 50,
        height: 50,
        borderRadius: 25,
    },
    itemMeta: {
        marginLeft: 10,
        justifyContent: 'center',
    },
    itemName: {
        fontSize: 20,
    },
    itemLastMessage: {
        fontSize: 14,
        color: "#111",
    }
});