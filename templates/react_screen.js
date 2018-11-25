import React, { Component } from 'react'
import { Text, View, ToolbarAndroid, StyleSheet, Image, FlatList } from 'react-native'

export default class ${filename} extends Component {

    static navigationOptions = { header: null }

    render() {
        return (
            <View style={{ marginTop: 24 }}>
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
});